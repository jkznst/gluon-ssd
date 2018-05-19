import logging
import time
import math
import os
import argparse

import numpy as np
from torchnet.logger import VisdomPlotLogger
import matplotlib.pyplot as plt

from mxnet import nd
from mxnet.contrib.ndarray import MultiBoxPrior, MultiBoxTarget, MultiBoxDetection
# from mxnet.contrib.symbol import MultiBoxPrior
from mxnet.gluon import nn
from mxnet import gluon
from mxnet import metric
from mxnet import gpu
from mxnet import autograd
import mxnet as mx

from config.config import cfg
from dataset.iterator import Det6DRecordIter


def get_scales(min_scale=0.2, max_scale=0.9,num_layers=6):
    """ Following the ssd arxiv paper, regarding the calculation of scales & ratios

    Parameters
    ----------
    min_scale : float
    max_scales: float
    num_layers: int
        number of layers that will have a detection head
    anchor_ratios: list
    first_layer_ratios: list

    return
    ------
    sizes : list
        list of scale sizes per feature layer
    ratios : list
        list of anchor_ratios per feature layer
    """

    # this code follows the original implementation of wei liu
    # for more, look at ssd/score_ssd_pascal.py:310 in the original caffe implementation
    min_ratio = int(min_scale * 100)
    max_ratio = int(max_scale * 100)
    step = int(np.floor((max_ratio - min_ratio) / (num_layers - 2)))
    min_sizes = []
    max_sizes = []
    for ratio in range(min_ratio, max_ratio + 1, step):
        min_sizes.append(ratio / 100.)
        max_sizes.append((ratio + step) / 100.)
    min_sizes = [int(100*min_scale / 2.0) / 100.0] + min_sizes
    max_sizes = [min_scale] + max_sizes

    # convert it back to this implementation's notation:
    scales = []
    for layer_idx in range(num_layers):
        scales.append([min_sizes[layer_idx], np.single(np.sqrt(min_sizes[layer_idx] * max_sizes[layer_idx]))])
    return scales


def class_predictor(num_anchors, num_classes):
    """return a layer to predict classes"""
    out = nn.Conv2D(channels=num_anchors * (num_classes + 1), kernel_size=3, padding=1, strides=1)
    out.collect_params(select='.*bias').setattr('lr_mult', 2.0)
    return out


def box_predictor(num_anchors):
    """return a layer to predict delta locations"""
    out = nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1, strides=1)
    out.collect_params(select='.*bias').setattr('lr_mult', 2.0)
    return out


def down_sample(num_filters, stride, padding, prefix=''):
    """stack two Conv-BatchNorm-Relu blocks
    to halve the feature size"""
    out = nn.HybridSequential(prefix=prefix)

    with out.name_scope():
        out.add(nn.Conv2D(num_filters // 2, kernel_size=1, strides=1, padding=0, prefix='conv_1x1_conv_'))
        out.add(nn.Activation('relu', prefix='conv_1x1_relu_'))
        out.add(nn.Conv2D(num_filters, 3, strides=stride, padding=padding, prefix='conv_3x3_conv_'))
        out.add(nn.Activation('relu', prefix='conv_3x3_relu_'))

    out.collect_params(select='.*bias').setattr('lr_mult', 2.0)
    return out


def training_targets(anchors, class_preds, labels):
    '''
    :param anchors:     1 x num_anchors x 4
    :param class_preds: batchsize x num_anchors x num_cls
    :param labels:      batchsize x label_width
    :return:
        box_target:     batchsize x (num_anchors x 4)
        box_mask:       batchsize x (num_anchors x 4)
        cls_target:     batchsize x num_anchors
    '''
    class_preds = class_preds.transpose(axes=(0, 2, 1)) # batchsize x num_cls x num_anchors
    box_target, box_mask, cls_target = MultiBoxTarget(anchors, labels, class_preds, overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")

    return box_target, box_mask, cls_target


def flatten_prediction(pred):
    return pred.transpose(axes=(0,2,3,1)).flatten()


def concat_predictions(preds):
    return nd.concat(*preds, dim=1)


class NormScale(gluon.HybridBlock):
    def __init__(self, channel, scale, prefix='scale_'):
        super(NormScale, self).__init__(prefix=prefix)
        with self.name_scope():
            self._channel = channel
            self._scale = scale
            self.weight = self.params.get('scale', shape=(1, channel, 1, 1),
                                          init=mx.init.Constant(scale), wd_mult=0.1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = F.L2Normalization(x, mode="channel")
        x = F.broadcast_mul(lhs=x, rhs=self.weight.data())
        return x


class SSD(gluon.HybridBlock):
    def __init__(self, network, data_shape, num_classes=1, num_view_classes=337, num_inplane_classes=18, verbose=False, **kwargs):
        super(SSD, self).__init__(prefix='ssd_', **kwargs)
        if network == 'vgg16_reduced':
            if data_shape >= 448:
                self.from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
                self.num_filters = [512, -1, 512, 256, 256, 256, 256]
                self.strides = [-1, -1, 2, 2, 2, 2, 1]
                self.pads = [-1, -1, 1, 1, 1, 1, 1]
                self.sizes = get_scales(min_scale=0.15, max_scale=0.9, num_layers=len(self.from_layers))
                self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                          [1, 2, .5, 3, 1. / 3], [1, 2, .5], [1, 2, .5]]
                self.normalizations = [20, -1, -1, -1, -1, -1, -1]
                self.normscale = NormScale(channel=self.num_filters[0], scale=self.normalizations[0], prefix='ssd_relu4_3_norm_')
                self.steps = [] if data_shape != 512 else [x / 512.0 for x in
                                                      [8, 16, 32, 64, 128, 256, 512]]
            else:
                self.from_layers = ['relu4_3', 'relu7', '', '', '', '']
                self.num_filters = [512, -1, 512, 256, 256, 256]
                self.strides = [-1, -1, 2, 2, 1, 1]
                self.pads = [-1, -1, 1, 1, 0, 0]
                self.sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(self.from_layers))
                self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                          [1, 2, .5], [1, 2, .5]]
                self.normalizations = [20, -1, -1, -1, -1, -1]
                self.normscale = NormScale(channel=self.num_filters[0], scale=self.normalizations[0], prefix='ssd_relu4_3_norm_')
                self.steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
            if not (data_shape == 300 or data_shape == 512):
                raise NotImplementedError("No implementation for shape: " + data_shape)
        # elif network == 'inceptionv3':
        #     if data_shape >= 448:
        #         self.from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
        #         self.num_filters = [-1, -1, 512, 256, 256, 128]
        #         self.strides = [-1, -1, 2, 2, 2, 2]
        #         self.pads = [-1, -1, 1, 1, 1, 1]
        #         self.sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(self.from_layers))
        #         self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
        #                   [1, 2, .5], [1, 2, .5]]
        #         self.normalizations = -1
        #         self.steps = []
        #     else:
        #         self.from_layers = ['ch_concat_mixed_2_chconcat', 'ch_concat_mixed_7_chconcat',
        #                        'ch_concat_mixed_10_chconcat', '', '', '']
        #         self.num_filters = [-1, -1, -1, 256, 256, 128]
        #         self.strides = [-1, -1, -1, 2, 2, 2]
        #         self.pads = [-1, -1, -1, 1, 1, 1]
        #         self.sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(self.from_layers))
        #         self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
        #                   [1, 2, .5], [1, 2, .5]]
        #         self.normalizations = -1
        #         self.steps = []
        else:
            raise NotImplementedError("No implementation for network: " + network)
        # anchor box sizes and ratios for 6 feature scales
        self.network = network
        self.num_classes = num_classes
        self.data_shape = data_shape
        self.num_view_classes = num_view_classes
        self.num_inplane_classes = num_inplane_classes

        self.verbose = verbose
        self.num_anchors = [len(s) + len(r) - 1 for s, r in zip(self.sizes, self.ratios)]
        # use name_scope to guard the names
        with self.name_scope():
            self.model = self.ssd_6d_model()

    def ssd_6d_model(self):
        # arguments check
        assert len(self.from_layers) > 0
        assert isinstance(self.from_layers[0], str) and len(self.from_layers[0].strip()) > 0
        assert len(self.from_layers) == len(self.num_filters) == len(self.strides) == len(self.pads)

        if self.network == 'vgg16_reduced':
            multifeatures = nn.HybridSequential(prefix='multi_feat_')
            with multifeatures.name_scope():
                vgg16_reduced = VGG16_reduced(self.num_classes)
            for k, params in enumerate(zip(self.from_layers, self.num_filters, self.strides, self.pads)):
                from_layer, num_filter, s, p = params
                if from_layer.strip():
                    with multifeatures.name_scope():
                        multifeatures.add(vgg16_reduced.dict[from_layer])
                if not from_layer.strip():
                    # attach from last feature layer
                    with multifeatures.name_scope():
                        multifeatures.add(down_sample(num_filters=num_filter, stride=s, padding=p, prefix='{}_'.format(k)))

            class_predictors = nn.HybridSequential(prefix="class_preds_")
            box_predictors = nn.HybridSequential(prefix="box_preds_")
            for i in range(len(multifeatures)):
                with class_predictors.name_scope():
                    class_predictors.add(class_predictor(self.num_anchors[i], self.num_classes))
                with box_predictors.name_scope():
                    box_predictors.add(box_predictor(self.num_anchors[i]))
            model = nn.HybridSequential(prefix="")
            with model.name_scope():
                model.add(multifeatures, class_predictors, box_predictors)
            return model
        # elif self.network == 'inceptionv3':
        #     multifeatures = nn.HybridSequential(prefix='multifeatures_')
        #     with multifeatures.name_scope():
        #         vgg16_reduced = VGG16_reduced(self.num_classes)
        #     for k, params in enumerate(zip(self.from_layers, self.num_filters, self.strides, self.pads)):
        #         from_layer, num_filter, s, p = params
        #         if from_layer.strip():
        #             with multifeatures.name_scope():
        #                 multifeatures.add(vgg16_reduced.dict[from_layer])
        #         if not from_layer.strip():
        #             # attach from last feature layer
        #             with multifeatures.name_scope():
        #                 multifeatures.add(
        #                     down_sample(num_filters=num_filter, stride=s, padding=p, prefix='{}_'.format(k)))
        #
        #     class_predictors = nn.HybridSequential(prefix="class_preds_")
        #     box_predictors = nn.HybridSequential(prefix="box_preds_")
        #     for i in range(len(multifeatures)):
        #         with class_predictors.name_scope():
        #             class_predictors.add(class_predictor(self.num_anchors[i], self.num_classes))
        #         with box_predictors.name_scope():
        #             box_predictors.add(box_predictor(self.num_anchors[i]))
        #     model = nn.HybridSequential(prefix="")
        #     with model.name_scope():
        #         model.add(multifeatures, class_predictors, box_predictors)
        #     return model
        else:
            raise NotImplementedError("No implementation for network: " + self.network)

    def ssd_6d_forward(self, x, model, sizes, ratios, steps=[], verbose=False):
        multifeatures, class_predictors, box_predictors = model

        anchors, class_preds, box_preds = [], [], []

        for i in range(len(multifeatures)):
            x = multifeatures[i](x)
            # predict
            # create anchor generation layer
            if steps:
                step = (steps[i], steps[i])
            else:
                step = '(-1.0, -1.0)'

            anchors.append(MultiBoxPrior(
                x, sizes=sizes[i], ratios=ratios[i], clip=False, steps=step))
            class_preds.append(
                flatten_prediction(class_predictors[i](x)))
            box_preds.append(
                flatten_prediction(box_predictors[i](x)))

            if verbose:
                print('Predict scale', i, x.shape, 'with',
                      anchors[-1].shape[1], 'anchors')

        # concat data
        return (concat_predictions(anchors),
                concat_predictions(class_preds),
                concat_predictions(box_preds))
        # concat_predictions(bb8_preds))

    def params_init(self, ctx):
        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
        self.normscale.collect_params().initialize(ctx=ctx)
        self.init_body_params(ctx)

    def init_body_params(self, ctx):
        if self.network == 'vgg16_reduced':
            self.model[0][0].load_params(filename="./models/vgg16_reduced-0001.params",
                                          ctx=ctx, allow_missing=True, ignore_extra=True)
            self.model[0][1].load_params(filename="./models/vgg16_reduced-0001.params",
                                          ctx=ctx, allow_missing=False, ignore_extra=True)
        else:
            raise NotImplementedError("No implementation for initialization of network: " + self.network)

    def hybrid_forward(self, F, x, *args, **kwargs):
        multifeatures, class_predictors, box_predictors = self.model

        anchors, class_preds, box_preds = [], [], []

        for i in range(len(multifeatures)):
            x = multifeatures[i](x)
            # normalize
            # if self.normalizations[i] > 0:
            #     x = F.L2Normalization(data=x, mode="channel")
                # scale = F.ones(shape=(1, self.num_filters[i], 1, 1)) * 1.0 #self.normalizations[i]
                # scale = gluon.Parameter(name="{}_scale".format('relu4_3'), grad_req='write',
                #                         shape=(1, self.num_filters[i], 1, 1), lr_mult=1.0,
                #                         wd_mult=0.1, init=mx.init.Constant(self.normalizations[i]))
                # scale.initialize(ctx=ctx)

                # x = F.broadcast_mul(lhs=scale.data(ctx), rhs=x)

            # predict
            if self.steps:
                step = (self.steps[i], self.steps[i])
            else:
                step = '(-1.0, -1.0)'

            anchors.append(MultiBoxPrior(
                x, sizes=self.sizes[i], ratios=self.ratios[i], clip=False, steps=step))
            if self.normalizations[i] > 0:
                class_preds.append(
                    flatten_prediction(class_predictors[i](self.normscale(x))))
                box_preds.append(
                    flatten_prediction(box_predictors[i](self.normscale(x))))
            else:
                class_preds.append(
                    flatten_prediction(class_predictors[i](x)))
                box_preds.append(
                    flatten_prediction(box_predictors[i](x)))

            if self.verbose:
                print('Predict scale', i, x.shape, 'with',
                      anchors[-1].shape[1], 'anchors')

        # concat data
        anchors = F.concat(*anchors, dim=1)
        class_preds = F.concat(*class_preds, dim=1)
        box_preds = F.concat(*box_preds, dim=1)

        # it is better to have class predictions reshaped for softmax computation
        class_preds = class_preds.reshape(shape=(0, -1, self.num_classes+1))

        return anchors, class_preds, box_preds


class FocalLoss(gluon.loss.Loss):
    def __init__(self, class_axis=-1, alpha=0.25, gamma=2, batch_axis=0, name='focalloss', **kwargs):
        super(FocalLoss, self).__init__(None, batch_axis, **kwargs)
        self._class_axis = class_axis
        self._alpha = alpha
        self._gamma = gamma
        self._name = name

    def hybrid_forward(self, F, pred, label, mask):
        '''
        :param F:
        :param pred:    batchsize x num_anchors x num_class
        :param label:   batchsize x num_anchors
        :param mask:    batchsize x num_anchors
        :return:
            masked focal loss
        '''
        pred = F.softmax(pred, axis=self._class_axis)
        pj = pred.pick(label, axis=self._class_axis) # batchsize x num_anchors
        loss = - self._alpha * ((1 - pj) ** self._gamma) * pj.log()
        loss = loss * mask
        output = F.sum(loss, self._batch_axis, exclude=True) / F.sum(mask, axis=self._batch_axis, exclude=True)
        return output


class MaskSoftmaxCELoss(gluon.loss.Loss):
    def __init__(self, class_axis=-1, batch_axis=0, name='maskSCEloss', **kwargs):
        super(MaskSoftmaxCELoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = class_axis
        self._name = name

    def hybrid_forward(self, F, pred, label, mask):
        '''
        :param F:
        :param pred:    batchsize x num_anchors x num_class
        :param label:   batchsize x num_anchors
        :param mask:    batchsize x num_anchors
        :return:
            masked cross entropy loss
        '''

        pred = F.log_softmax(pred, axis=self._axis)
        loss = -F.pick(pred, label, axis=self._axis)
        loss = loss * mask
        output = F.sum(loss, axis=self._batch_axis, exclude=True)
        num_mask = F.sum(mask, axis=self._batch_axis, exclude=True)
        output = output / num_mask
        return output


class SmoothL1Loss(gluon.loss.Loss):
    def __init__(self, batch_axis=0, name='smoothl1loss', **kwargs):
        super(SmoothL1Loss, self).__init__(None, batch_axis, **kwargs)
        self._name = name

    def hybrid_forward(self, F, pred, label, mask):
        '''
        :param F:
        :param pred:    batchsize x (num_anchors x 4)
        :param label:   batchsize x (num_anchors x 4)
        :param mask:    batchsize x (num_anchors x 4)
        :return:
            masked smooth_l1 loss
        '''
        loss = F.smooth_l1((pred - label) * mask, scalar=1.0)
        output = F.sum(loss, self._batch_axis, exclude=True) / F.sum(mask, axis=self._batch_axis, exclude=True)
        return output


class MaskMAE(mx.metric.EvalMetric):
    """
    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self, name='maskmae',
                 output_names=None, label_names=None):
        super(MaskMAE, self).__init__(
            name, output_names=output_names, label_names=label_names)

    def update(self, labels, preds, masks):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.

        masks : list of 'NDArray'

        """
        mx.metric.check_label_shapes(labels, preds)

        for label, pred, mask in zip(labels, preds, masks):
            label = label.asnumpy()
            pred = pred.asnumpy()
            mask = mask.asnumpy()

            if len(label.shape) == 1:
                label = label.reshape(label.shape[0], 1)

            self.sum_metric += np.abs(label - pred * mask).sum() / mask.sum()
            self.num_inst += 4  # numpy.prod(label.shape)


class MApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    ovp_thresh : float
        overlap threshold for TP
    use_difficult : boolean
        use difficult ground-truths if applicable, otherwise just ignore
    class_names : list of str
        optional, if provided, will print out AP for each class
    pred_idx : int
        prediction index in network output list
    roc_output_path
        optional, if provided, will save a ROC graph for each class
    tensorboard_path
        optional, if provided, will save a ROC graph to tensorboard
    """
    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None,
                 pred_idx=0, roc_output_path=None, tensorboard_path=None):
        super(MApMetric, self).__init__('mAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = class_names + ['mAP']
            self.num = num + 1
        self.reset()
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)
        self.roc_output_path = roc_output_path
        self.tensorboard_path = tensorboard_path

    def save_roc_graph(self, recall=None, prec=None, classkey=1, path=None, ap=None):
        if not os.path.exists(path):
            os.mkdir(path)
        plot_path = os.path.join(path, 'roc_'+self.class_names[classkey])
        if os.path.exists(plot_path):
            os.remove(plot_path)
        fig = plt.figure()
        plt.title(self.class_names[classkey])
        plt.plot(recall, prec, 'b', label='AP = %0.2f' % ap)
        plt.legend(loc='lower right')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.savefig(plot_path)
        plt.close(fig)

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self.records = dict()
        self.counts = dict()

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan') \
                for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def update(self, labels, preds):
        """
        Update internal records. This function now only update internal buffer,
        sum_metric and num_inst are updated in _update() function instead when
        get() is called to return results.

        Params:
        ----------
        labels: mx.nd.array (n * 6) or (n * 5), difficult column is optional
            4-d array of ground-truths, n objects(id-xmin-ymin-xmax-ymax-[difficult])
        preds: mx.nd.array (m * 6)
            4-d array of detections, m objects(id-score-xmin-ymin-xmax-ymax)
        """
        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) * \
                (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        # independant execution for each image
        for i in range(labels[0].shape[0]):
            # get as numpy arrays
            label = labels[0][i].asnumpy()
            pred = preds[self.pred_idx][i].asnumpy()
            # calculate for each class
            while (pred.shape[0] > 0):
                cid = int(pred[0, 0])
                indices = np.where(pred[:, 0].astype(int) == cid)[0]
                if cid < 0:
                    pred = np.delete(pred, indices, axis=0)
                    continue
                dets = pred[indices]
                pred = np.delete(pred, indices, axis=0)
                # sort by score, desceding
                dets[dets[:,1].argsort()[::-1]]
                records = np.hstack((dets[:, 1][:, np.newaxis], np.zeros((dets.shape[0], 1))))
                # ground-truths
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                gts = label[label_indices, :]
                label = np.delete(label, label_indices, axis=0)
                if gts.size > 0:
                    found = [False] * gts.shape[0]
                    for j in range(dets.shape[0]):
                        # compute overlaps
                        ious = iou(dets[j, 2:], gts[:, 1:5])
                        ovargmax = np.argmax(ious)
                        ovmax = ious[ovargmax]
                        if ovmax > self.ovp_thresh:
                            if (not self.use_difficult and
                                gts.shape[1] >= 6 and
                                gts[ovargmax, 5] > 0):
                                pass
                            else:
                                if not found[ovargmax]:
                                    records[j, -1] = 1  # tp
                                    found[ovargmax] = True
                                else:
                                    # duplicate
                                    records[j, -1] = 2  # fp
                        else:
                            records[j, -1] = 2 # fp
                else:
                    # no gt, mark all fp
                    records[:, -1] = 2

                # ground truth count
                if (not self.use_difficult and gts.shape[1] >= 6):
                    gt_count = np.sum(gts[:, 5] < 1)
                else:
                    gt_count = gts.shape[0]

                # now we push records to buffer
                # first column: score, second column: tp/fp
                # 0: not set(matched to difficult or something), 1: tp, 2: fp
                records = records[np.where(records[:, -1] > 0)[0], :]
                if records.size > 0:
                    self._insert(cid, records, gt_count)

            # add missing class if not present in prediction
            while (label.shape[0] > 0):
                cid = int(label[0, 0])
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                label = np.delete(label, label_indices, axis=0)
                if cid < 0:
                    continue
                gt_count = label_indices.size
                self._insert(cid, np.array([[0, 0]]), gt_count)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        for k, v in self.records.items():
            recall, prec = self._recall_prec(v, self.counts[k])
            ap = self._average_precision(recall, prec)
            if self.roc_output_path is not None:
                self.save_roc_graph(recall=recall, prec=prec, classkey=k, path=self.roc_output_path, ap=ap)
            aps.append(ap)
            if self.num is not None and k < (self.num - 1):
                self.sum_metric[k] = ap
                self.num_inst[k] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.mean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.mean(aps)

    def _recall_prec(self, record, count):
        """ get recall and precision from internal records """
        record = np.delete(record, np.where(record[:, 1].astype(int) == 0)[0], axis=0)
        sorted_records = record[record[:,0].argsort()[::-1]]
        tp = np.cumsum(sorted_records[:, 1].astype(int) == 1)
        fp = np.cumsum(sorted_records[:, 1].astype(int) == 2)
        if count <= 0:
            recall = tp * 0.0
        else:
            recall = tp / float(count)
        prec = tp.astype(float) / (tp + fp)
        return recall, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _insert(self, key, records, count):
        """ Insert records according to key """
        if key not in self.records:
            assert key not in self.counts
            self.records[key] = records
            self.counts[key] = count
        else:
            self.records[key] = np.vstack((self.records[key], records))
            assert key in self.counts
            self.counts[key] += count


class VOC07MApMetric(MApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset """
    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap


def get_optimizer_params(optimizer=None, learning_rate=None, momentum=None,
                         weight_decay=None, lr_scheduler=None, ctx=None, logger=None):
    if optimizer.lower() == 'rmsprop':
        opt = 'rmsprop'
        logger.info('you chose RMSProp, decreasing lr by a factor of 10')
        optimizer_params = {'learning_rate': learning_rate / 10.0,
                            'wd': weight_decay,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0}
    elif optimizer.lower() == 'sgd':
        opt = 'sgd'
        optimizer_params = {'learning_rate': learning_rate,
                            'momentum': momentum,
                            'wd': weight_decay,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0}
    elif optimizer.lower() == 'adadelta':
        opt = 'adadelta'
        optimizer_params = {}
    elif optimizer.lower() == 'adam':
        opt = 'adam'
        optimizer_params = {'learning_rate': learning_rate,
                            'lr_scheduler': lr_scheduler,
                            'clip_gradient': None,
                            'rescale_grad': 1.0}
    return opt, optimizer_params


def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            pass
            # logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)


class Residual(nn.HybridBlock):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1,
                                  strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1,
                                      strides=strides)

    def hybrid_forward(self, F, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3(x)
        return F.relu(out + x)


class ResNet(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            net = self.net = nn.HybridSequential()
            # group 1
            net.add(nn.Conv2D(channels=32, kernel_size=3, strides=1, padding=1))
            net.add(nn.BatchNorm())
            net.add(nn.Activation(activation='relu'))
            # group 2
            for _ in range(3):
                net.add(Residual(channels=32))
            # group 3
            net.add(Residual(channels=64, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=64))
            # group 4
            net.add(Residual(channels=128, same_shape=False))
            for _ in range(2):
                net.add(Residual(channels=128))
            # group 5
            net.add(nn.AvgPool2D(pool_size=8))
            net.add(nn.Flatten())
            net.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output: %s'%(i+1, out.shape))
        return out


class VGG16_reduced(nn.HybridBlock):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(VGG16_reduced, self).__init__(prefix='vgg16_reduced_', **kwargs)
        self.verbose = verbose
        with self.name_scope():
            self.relu4_3 = nn.HybridSequential(prefix='relu4_3_')
            with self.relu4_3.name_scope():
                # group 1
                self.relu4_3.add(
                    nn.Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1), prefix="conv1_1_"),
                    nn.Activation(activation="relu", prefix="relu1_1_"),
                    nn.Conv2D(channels=64, kernel_size=(3, 3), padding=(1, 1), prefix="conv1_2_"),
                    nn.Activation(activation="relu", prefix="relu1_2_"),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), prefix="pool1_")
                )
                # group 2
                self.relu4_3.add(
                    nn.Conv2D(channels=128, kernel_size=3, strides=1, padding=1, prefix="conv2_1_"),
                    nn.Activation(activation="relu", prefix="relu2_1_"),
                    nn.Conv2D(channels=128, kernel_size=3, strides=1, padding=1, prefix="conv2_2_"),
                    nn.Activation(activation="relu", prefix="relu2_2_"),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), prefix="pool2_")
                )
                # group 3
                self.relu4_3.add(
                    nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, prefix="conv3_1_"),
                    nn.Activation(activation="relu", prefix="relu3_1_"),
                    nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, prefix="conv3_2_"),
                    nn.Activation(activation="relu", prefix="relu3_2_"),
                    nn.Conv2D(channels=256, kernel_size=3, strides=1, padding=1, prefix="conv3_3_"),
                    nn.Activation(activation="relu", prefix="relu3_3_"),
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), ceil_mode=True, prefix="pool3_")
                )
                # group 4
                self.relu4_3.add(
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv4_1_"),
                    nn.Activation(activation="relu", prefix="relu4_1_"),
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv4_2_"),
                    nn.Activation(activation="relu", prefix="relu4_2_"),
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv4_3_"),
                    nn.Activation(activation="relu", prefix="relu4_3_"),
                )

            self.relu7 = nn.HybridSequential(prefix="relu7_")
            with self.relu7.name_scope():
                self.relu7.add(
                    # self.relu4_3,
                    nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2), prefix="pool4_")
                )
                # group 5
                self.relu7.add(
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv5_1_"),
                    nn.Activation(activation="relu", prefix="relu4_1_"),
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv5_2_"),
                    nn.Activation(activation="relu", prefix="relu4_2_"),
                    nn.Conv2D(channels=512, kernel_size=3, strides=1, padding=1, prefix="conv5_3_"),
                    nn.Activation(activation="relu", prefix="relu4_3_"),
                    nn.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding=(1, 1), prefix="pool5_")
                )
                # group 6
                self.relu7.add(
                    nn.Conv2D(channels=1024, kernel_size=(3, 3), strides=(1, 1), padding=(6, 6),
                              dilation=(6, 6), prefix="fc6_"),
                    nn.Activation(activation="relu", prefix="relu6_")
                    # nn.Dropout(rate=0.5, prefix="drop6")
                )
                # group 7
                self.relu7.add(
                    nn.Conv2D(channels=1024, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), prefix="fc7_"),
                    nn.Activation(activation="relu", prefix="relu7_")
                    # nn.Dropout(rate=0.5, prefix="drop7")
                )

            self.whole_net = nn.HybridSequential(prefix="whole_net_")
            with self.whole_net.name_scope():
                # group 8
                self.whole_net.add(
                    self.relu4_3,
                    self.relu7,
                    nn.GlobalAvgPool2D(prefix="global_pool_"),
                    nn.Conv2D(channels=num_classes, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0), prefix="fc8_"),
                    nn.Flatten()
                )
        self.dict = {'relu4_3' : self.relu4_3,
                     'relu7' : self.relu7}

    def get_symbol(self):
        self.whole_net.hybridize()
        x = mx.sym.var('data')
        y = self.whole_net(x)
        return y


# class Inceptionv3(nn.HybridBlock):
#     def __init__(self, num_classes, verbose=False, **kwargs):
#         super(Inceptionv3, self).__init__(prefix='inceptionv3_', **kwargs)
#         self.verbose = verbose
#         self.num_class = num_classes
#
#     def Conv(self, data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix=''):
#         conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, no_bias=True,
#                                   name='%s%s_conv2d' % (name, suffix))
#         bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' % (name, suffix), fix_gamma=True)
#         act = mx.sym.Activation(data=bn, act_type='relu', name='%s%s_relu' % (name, suffix))
#         return act
#
#     def get_symbol(self):
#         self.whole_net.hybridize()
#         x = mx.sym.var('data')
#         y = self.whole_net(x)
#         self.dict = {'relu4_3': self.relu4_3,
#                      'relu7': self.relu7}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Single-shot detection network')
    parser.add_argument('--train-path', dest='train_path', help='train record to use',
                        default=os.path.join(os.getcwd(), 'data', 'VOC0712rec', 'train.rec'), type=str)
    parser.add_argument('--train-list', dest='train_list', help='train list to use',
                        default="", type=str)
    parser.add_argument('--val-path', dest='val_path', help='validation record to use',
                        default=os.path.join(os.getcwd(), 'data', 'VOC0712rec', 'val.rec'), type=str)
    parser.add_argument('--val-list', dest='val_list', help='validation list to use',
                        default="", type=str)
    parser.add_argument('--network', dest='network', type=str, default='vgg16_reduced',
                        help='which network to use')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='training batch size')
    parser.add_argument('--resume', dest='resume', type=int, default=-1,
                        help='resume training from epoch n')
    parser.add_argument('--finetune', dest='finetune', type=int, default=-1,
                        help='finetune from epoch n, rename the model before doing this')
    parser.add_argument('--pretrained', dest='pretrained', help='pretrained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'), type=str)
    parser.add_argument('--epoch', dest='epoch', help='epoch of pretrained model',
                        default=1, type=int)
    parser.add_argument('--prefix', dest='prefix', help='new model prefix',
                        default=os.path.join(os.getcwd(), 'output', 'exp1', 'ssd'), type=str)
    parser.add_argument('--gpus', dest='gpus', help='GPU devices to train with',
                        default='0', type=str)
    parser.add_argument('--begin-epoch', dest='begin_epoch', help='begin epoch of training',
                        default=0, type=int)
    parser.add_argument('--end-epoch', dest='end_epoch', help='end epoch of training',
                        default=240, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=20, type=int)
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=300,
                        help='set image shape')
    parser.add_argument('--label-width', dest='label_width', type=int, default=350,
                        help='force padding label width to sync across train and validation')
    parser.add_argument('--optimizer', dest='optimizer', type=str, default='sgd',
                        help='Whether to use a different optimizer or follow the original code with sgd')
    parser.add_argument('--lr', dest='learning_rate', type=float, default=0.004,
                        help='learning rate')
    parser.add_argument('--momentum', dest='momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', dest='weight_decay', type=float, default=0.0005,
                        help='weight decay')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--lr-steps', dest='lr_refactor_step', type=str, default='80, 160',
                        help='refactor learning rate at specified epochs')
    parser.add_argument('--lr-factor', dest='lr_refactor_ratio', type=str, default=0.1,
                        help='ratio to refactor learning rate')
    parser.add_argument('--freeze', dest='freeze_pattern', type=str, default="^(conv1_|conv2_).*",
                        help='freeze layer pattern')
    parser.add_argument('--log', dest='log_file', type=str, default="train.log",
                        help='save training log to file')
    parser.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    parser.add_argument('--pattern', dest='monitor_pattern', type=str, default=".*",
                        help='monitor parameter pattern, as regex')
    parser.add_argument('--num-class', dest='num_class', type=int, default=20,
                        help='number of classes')
    parser.add_argument('--num-example', dest='num_example', type=int, default=16551,
                        help='number of image examples')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.45,
                        help='non-maximum suppression threshold')
    parser.add_argument('--nms_topk', dest='nms_topk', type=int, default=400,
                        help='final number of detections')
    parser.add_argument('--overlap', dest='overlap_thresh', type=float, default=0.5,
                        help='evaluation overlap threshold')
    parser.add_argument('--force', dest='force_nms', type=bool, default=False,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--use-difficult', dest='use_difficult', type=bool, default=False,
                        help='use difficult ground-truths in evaluation')
    parser.add_argument('--voc07', dest='use_voc07_metric', type=bool, default=True,
                        help='use PASCAL VOC 07 11-point metric')
    parser.add_argument('--tensorboard', dest='tensorboard', type=bool, default=False,
                        help='save metrics into tensorboard readable files')
    parser.add_argument('--min_neg_samples', dest='min_neg_samples', type=int, default=0,
                        help='min number of negative samples taken in hard mining.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    data_shape = (3, args.data_shape, args.data_shape)
    batch_size = args.batch_size
    mean_pixels = [args.mean_r, args.mean_g, args.mean_b]

    num_samples = args.num_example

    label_pad_width = args.label_width
    colors = ['blue', 'green', 'red', 'black', 'magenta']
    train_path = args.train_path
    train_list = args.train_list
    val_path = args.val_path
    val_list = args.val_list

    if os.path.exists(train_path.replace('.rec','.idx')):
        with open(train_path.replace('.rec','.idx'), 'r') as f:
            txt = f.readlines()
        num_samples = len(txt)

    num_batches = math.ceil(num_samples / batch_size)
    ctx = mx.gpu(int(args.gpus))
    checkpoint_period = 10
    use_visdom = True
    log_file = 'train.log'
    prefix = os.path.join(os.getcwd(), 'output', 'exp1', 'ssd')
    class_name = 'aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, ' \
                 'person, pottedplant, sheep, sofa, train, tvmonitor'
    class_name = class_name.split(', ')

    # export mxnet pretrained models
    # data = mx.nd.ones(shape=(1,3,224,224))
    # net = gluon.model_zoo.vision.resnet50_v2(pretrained=True)
    # net.hybridize()
    # out = net(data)
    # net.export(path='./', epoch=0)

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        log_file_path = os.path.join(os.path.dirname(prefix), log_file)
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path))
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)

    train_iter = Det6DRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
                                   label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)

    if val_path:
        val_iter = Det6DRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
                                     label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
    else:
        val_iter = None
    # val_iter = None


    cls_loss = MaskSoftmaxCELoss(class_axis=-1, batch_axis=0)
    # cls_loss = FocalLoss(class_axis=-1, batch_axis=0, alpha=0.25, gamma=2)
    box_loss = SmoothL1Loss(batch_axis=0)

    cls_metric = mx.metric.Accuracy(axis=-1)
    box_metric = MaskMAE()
    val_metric = VOC07MApMetric(ovp_thresh=0.45, class_names=class_name, roc_output_path=os.path.join(os.path.dirname(prefix), 'roc'))

    net = SSD(network=args.network, data_shape=args.data_shape, num_classes=20, verbose=False)
    net.params_init(ctx)
    # net.hybridize()

    # freeze several layers
    net.collect_params('.*(vgg16_reduced_relu4_3_conv1_|vgg16_reduced_relu4_3_conv2_).*').setattr('grad_req', 'null')
    # finetune mode
    # net.collect_params('.*(bias)$').setattr('lr_mult', 2)
    # net.collect_params('.*(vgg16_reduced).*(bias)$').setattr('lr_mult', 1)
    # print(net.collect_params(select='.*(vgg16_reduced_relu4_3_conv1_|vgg16_reduced_relu4_3_conv2_).*'))

    # start visdom: $ sudo python3 -m visdom.server
    if use_visdom:
        train_cls_loss_logger = VisdomPlotLogger(
            'line', port=8097, opts={'title': 'Train Classification Loss'}
        )
        train_reg_loss_logger = VisdomPlotLogger(
            'line', port=8097, opts={'title': 'Train Regression Loss'}
        )
    cnt = 0

    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate=args.learning_rate, lr_refactor_ratio=0.1,
                                    lr_refactor_step='80, 160', num_example=num_samples, batch_size=batch_size, begin_epoch=0)
    # add possibility for different optimizer
    opt, opt_params = get_optimizer_params(optimizer=args.optimizer, learning_rate=learning_rate, momentum=0.9,
                                            weight_decay=5e-4, lr_scheduler=lr_scheduler, ctx=ctx)

    trainer = gluon.Trainer(net.collect_params(), opt, opt_params)

    logger.info('data_shape: {}, batch_size: {}, train_cls_loss: {}, train_bbox_loss: {}.'.format(
        data_shape, batch_size, cls_loss.prefix, box_loss.prefix
    ))
    logger.info('optimizer: {}, learning rate: {}, lr_refactor_ratio: 0.1, lr_refactor_step: 80,160'.format(
        args.optimizer, learning_rate))

    for epoch in range(240):
        # reset data iterators and metrics
        train_iter.reset()
        val_iter.reset()

        cls_metric.reset()
        box_metric.reset()
        val_metric.reset()
        tic = time.time()

        train_cls_loss = 0.0
        train_bbox_loss = 0.0
        cls_loss_list = list()
        reg_loss_list = list()

        for i, batch in enumerate(train_iter):
            btic = time.time()
            x = batch.data[0].as_in_context(ctx)  # 32 x 3 x 300 x 300
            y = batch.label[0].as_in_context(ctx)   # 32 x 43 x 8

            with autograd.record():
                '''
                # anchors:      1 x num_anchors x 4
                # class_preds:  batchsize x num_anchors x num_class
                # box_preds:    batchsize x (num_anchors x 4)
                '''
                anchors, class_preds, box_preds = net(x)

                with autograd.pause(train_mode=True):
                    '''
                    # box_target:   batchsize x (num_anchors x 4)
                    # box_mask:     batchsize x (num_anchors x 4)
                    # cls_target:   batchsize x num_anchors
                    '''
                    box_target, box_mask, cls_target = training_targets(anchors, class_preds, y)

                    cls_positive_mask = cls_target > 0
                    # cls_target_np = cls_target.asnumpy()
                    # cls_target_in_use = cls_target_np[cls_target_np.nonzero()]
                    # num_positive = mx.nd.sum(data=cls_positive_mask, axis=1, keepdims=False, exclude=False).asnumpy()
                    cls_positive_negative_mask = cls_target >= 0
                    # num_positive_negative = mx.nd.sum(data=cls_positive_negative_mask, axis=1, keepdims=False, exclude=False).asnumpy()
                # losses
                loss_cls = cls_loss(class_preds, cls_target, cls_positive_negative_mask)
                loss_bbox = box_loss(box_preds, box_target, box_mask)

                loss =  1.0 * loss_bbox + 1.0 * loss_cls

            cls_loss_list.append(nd.mean(loss_cls)[0].asscalar())
            reg_loss_list.append(nd.mean(loss_bbox)[0].asscalar())
            loss.backward()
            trainer.step(batch_size)

            # update metrics
            cls_metric.update(labels=[cls_target * cls_positive_mask], preds=[class_preds])
            box_metric.update(labels=[box_target], preds=[box_preds], masks=[box_mask])

            if (i + 1) % 20 == 0:
                val1 = np.mean(cls_loss_list)
                val2 = np.mean(reg_loss_list)

                logger.info('[Epoch %d Batch %d] speed: %f samples/s, training: %s=%f, %s=%f, %s=%f, %s=%f' % (
                    epoch, i, batch_size / (time.time() - btic), "cls loss", val1, "reg loss", val2,
                    "cls_metric", cls_metric.get()[1], "box metric", box_metric.get()[1]))
                if use_visdom:
                    train_cls_loss_logger.log(cnt, val1)
                    train_reg_loss_logger.log(cnt, val2)
                cls_loss_list = []
                reg_loss_list = []
                cnt += 1

            train_cls_loss += nd.mean(loss_cls).asscalar()
            train_bbox_loss += nd.mean(loss_bbox).asscalar()

            # print("train_cls_loss: %5f, train_box_loss %5f,"
            #       "time %.1f sec"
            #       % (nd.mean(loss_cls).asscalar(), nd.mean(loss_bbox).asscalar(), time.time() - tic))

        if epoch % checkpoint_period == 0:
            net.save_params(filename='output/exp1/ssd_{}_{}_epoch{}.params'.format(net.network,
                                                                              net.data_shape, epoch))

        if val_iter is not None:
            for i, batch in enumerate(val_iter):
                x = batch.data[0].as_in_context(ctx)  # 32 x 3 x 300 x 300
                y = batch.label[0].as_in_context(ctx)  # 32 x 43 x 8
                anchors, class_preds, box_preds = net(x)
                cls_probs = mx.nd.SoftmaxActivation(data=class_preds.transpose((0,2,1)), mode='channel')
                det = MultiBoxDetection(*[cls_probs, box_preds, anchors], \
                    name="detection", nms_threshold=0.45, force_suppress=False,
                    variances=(0.1, 0.1, 0.2, 0.2), nms_topk=400)

                val_metric.update(labels=[y], preds=[det])
            names, values = val_metric.get()
            # epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
            #              % (epoch, train_loss / len(train_data),
            #                 train_acc / len(train_data), valid_acc))
            for name, value in zip(names, values):
                logger.info('Epoch[{}] Validation-{}={}'.format(epoch, name, value))
        else:
            logger.info ("Epoch %2d. train_cls_loss: %5f, train_box_loss %5f,"
                         " time %.1f sec"
                         % (epoch, train_cls_loss / i, train_bbox_loss / i,
                             time.time()-tic))
            logger.info ('Epoch %2d, train %s %.2f, %s %.5f, time %.1f sec' % (
                epoch, *cls_metric.get(), *box_metric.get(), time.time() - tic
            ))