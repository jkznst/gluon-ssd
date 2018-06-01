from mxnet import gluon


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
