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
from dataset.iterator import DetRecordIter
from symbol.vgg16_reduced import VGG16_reduced, NormScale
from symbol.ResNet import ResNet
from loss import FocalLoss, MaskSoftmaxCELoss, SmoothL1Loss
from metric import MaskMAE, MApMetric, VOC07MApMetric
from optim import get_optimizer_params, get_lr_scheduler


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


class SSD(gluon.HybridBlock):
    def __init__(self, network, data_shape, num_classes=1, num_view_classes=337, num_inplane_classes=18, verbose=False, **kwargs):
        super(SSD, self).__init__(prefix='ssd_', **kwargs)
        if network == 'vgg16_reduced':
            if data_shape >= 448:
                self.network = network
                self.data_shape = data_shape
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
                self.network = network
                self.data_shape = data_shape
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
        elif network == 'resnet50':
            self.network = 'resnet'
            self.data_shape = '3,224,224'   # resnet require it as shape check
            self.num_layers = 50

            self.from_layers = ['_plus12', '_plus15', '', '', '', '']
            # from_layers = ['_plus6', '_plus12', '_plus15', '', '', '']
            self.num_filters = [-1, -1, 512, 256, 256, 128]
            self.strides = [-1, -1, 2, 2, 2, 2]
            self.pads = [-1, -1, 1, 1, 1, 1]
            self.sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(self.from_layers))
            self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                      [1, 2, .5], [1, 2, .5]]
            self.normalizations = [-1,-1,-1,-1,-1,-1]
            self.steps = []

        elif network == 'resnet101':
            self.network = 'resnet'
            self.data_shape = '3,224,224'
            self.num_layers = 101
            self.from_layers = ['_plus29', '_plus32', '', '', '', '']
            self.num_filters = [-1, -1, 512, 256, 256, 128]
            self.strides = [-1, -1, 2, 2, 2, 2]
            self.pads = [-1, -1, 1, 1, 1, 1]
            self.sizes = get_scales(min_scale=0.2, max_scale=0.9, num_layers=len(self.from_layers))
            self.ratios = [[1, 2, .5], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], [1, 2, .5, 3, 1. / 3], \
                      [1, 2, .5], [1, 2, .5]]
            self.normalizations = [-1,-1,-1,-1,-1,-1]
            self.steps = []

        else:
            raise NotImplementedError("No implementation for network: " + network)
        # anchor box sizes and ratios for 6 feature scales

        self.num_classes = num_classes

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
        elif self.network == 'resnet':
            (nchannel, height, width) = self.data_shape
            if height <= 28:
                num_stages = 3
                if (self.num_layers - 2) % 9 == 0 and self.num_layers >= 164:
                    per_unit = [(self.num_layers - 2) // 9]
                    filter_list = [16, 64, 128, 256]
                    bottle_neck = True
                elif (self.num_layers - 2) % 6 == 0 and self.num_layers < 164:
                    per_unit = [(self.num_layers - 2) // 6]
                    filter_list = [16, 16, 32, 64]
                    bottle_neck = False
                else:
                    raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(self.num_layers))
                units = per_unit * num_stages
            else:
                if self.num_layers >= 50:
                    filter_list = [64, 256, 512, 1024, 2048]
                    bottle_neck = True
                else:
                    filter_list = [64, 64, 128, 256, 512]
                    bottle_neck = False
                num_stages = 4
                if self.num_layers == 18:
                    units = [2, 2, 2, 2]
                elif self.num_layers == 34:
                    units = [3, 4, 6, 3]
                elif self.num_layers == 50:
                    units = [3, 4, 6, 3]
                elif self.num_layers == 101:
                    units = [3, 4, 23, 3]
                elif self.num_layers == 152:
                    units = [3, 8, 36, 3]
                elif self.num_layers == 200:
                    units = [3, 24, 36, 3]
                elif self.num_layers == 269:
                    units = [3, 30, 48, 8]
                else:
                    raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(self.num_layers))

            multifeatures = nn.HybridSequential(prefix='multi_feat_')
            with multifeatures.name_scope():
                resnet = ResNet(units = units,
                                num_stages  = num_stages,
                                filter_list = filter_list,
                                num_classes = self.num_classes,
                                image_shape = self.data_shape,
                                bottle_neck = bottle_neck
                                )
            for k, params in enumerate(zip(self.from_layers, self.num_filters, self.strides, self.pads)):
                from_layer, num_filter, s, p = params
                if from_layer.strip():
                    with multifeatures.name_scope():
                        multifeatures.add(resnet.dict[from_layer])
                if not from_layer.strip():
                    # attach from last feature layer
                    with multifeatures.name_scope():
                        multifeatures.add(
                            down_sample(num_filters=num_filter, stride=s, padding=p, prefix='{}_'.format(k)))

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
        else:
            raise NotImplementedError("No implementation for network: " + self.network)

    # def ssd_6d_forward(self, x, model, sizes, ratios, steps=[], verbose=False):
    #     multifeatures, class_predictors, box_predictors = model
    #
    #     anchors, class_preds, box_preds = [], [], []
    #
    #     for i in range(len(multifeatures)):
    #         x = multifeatures[i](x)
    #         # predict
    #         # create anchor generation layer
    #         if steps:
    #             step = (steps[i], steps[i])
    #         else:
    #             step = '(-1.0, -1.0)'
    #
    #         anchors.append(MultiBoxPrior(
    #             x, sizes=sizes[i], ratios=ratios[i], clip=False, steps=step))
    #         class_preds.append(
    #             flatten_prediction(class_predictors[i](x)))
    #         box_preds.append(
    #             flatten_prediction(box_predictors[i](x)))
    #
    #         if verbose:
    #             print('Predict scale', i, x.shape, 'with',
    #                   anchors[-1].shape[1], 'anchors')
    #
    #     # concat data
    #     return (concat_predictions(anchors),
    #             concat_predictions(class_preds),
    #             concat_predictions(box_preds))
    #     # concat_predictions(bb8_preds))

    def params_init(self, ctx):
        self.model.collect_params().initialize(mx.init.Xavier(magnitude=2), ctx=ctx)
        self.normscale.collect_params().initialize(ctx=ctx)
        self.init_body_params(ctx)

    def init_body_params(self, ctx):
        if self.network == 'vgg16_reduced':
            self.model[0][0].load_params(filename="./model/vgg16_reduced-0001.params",
                                          ctx=ctx, allow_missing=True, ignore_extra=True)
            self.model[0][1].load_params(filename="./model/vgg16_reduced-0001.params",
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
    log_file = args.log_file
    prefix = args.prefix
    class_name = [c.strip() for c in args.class_names.split(',')]

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

    train_iter = DetRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
                                   label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)

    if val_path:
        val_iter = DetRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
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

    net = SSD(network=args.network, data_shape=args.data_shape, num_classes=args.num_class, verbose=False)
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

    for epoch in range(args.end_epoch):
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