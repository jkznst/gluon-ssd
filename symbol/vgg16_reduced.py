from mxnet.gluon import nn
import mxnet as mx

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

        self.dict = {'relu4_3' : self.relu4_3,
                     'relu7' : self.relu7}


class NormScale(nn.HybridBlock):
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
