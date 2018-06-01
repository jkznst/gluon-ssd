from mxnet.gluon import nn

class residual_unit(nn.HybridBlock):
    def __init__(self, num_filter, stride, dim_match, prefix='', bottle_neck=True, bn_mom=0.9, **kwargs):
        """Return ResNet Unit symbol for building ResNet
        Parameters
        ----------
        num_filter : int
            Number of output channels
        bnf : int
            Bottle neck channels factor with regard to num_filter
        stride : tupe
            Stride used in convolution
        dim_match : Boolen
            True means channel number between input and output is the same, otherwise means differ
        name : str
            Base name of the operators
        workspace : int
            Workspace used in convolution operator
        """
        super(residual_unit, self).__init__(prefix=prefix, **kwargs)
        self.dim_match = dim_match
        out = nn.HybridSequential(prefix=prefix)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
            with self.name_scope():
                self.bn1 = nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn1_'),
                self.relu1 = nn.Activation(activation='relu', prefix='relu1_'),
                self.conv1 = nn.Conv2D(channels=int(num_filter*0.25), kernel_size=(1,1), strides=(1,1),
                          padding=(0,0), use_bias=False, prefix='conv1_'),
                self.bn2 = nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn2_'),
                self.relu2 = nn.Activation(activation='relu', prefix='relu2_'),
                self.conv2 = nn.Conv2D(channels=int(num_filter * 0.25), kernel_size=(3, 3), strides=stride,
                          padding=(1, 1), use_bias=False, prefix='conv2_'),
                self.bn3 = nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn3_'),
                self.relu3 = nn.Activation(activation='relu', prefix='relu3_'),
                self.conv3 = nn.Conv2D(channels=num_filter, kernel_size=(1, 1), strides=(1, 1),
                          padding=(0, 0), use_bias=False, prefix='conv3_')

                if not dim_match:
                    self.convsc = nn.Conv2D(channels=num_filter, kernel_size=(1,1), strides=stride, padding=(0,0),
                                            use_bias=False, prefix='sc_')

        else:
            with out.name_scope():
                self.bn1 = nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn1_'),
                self.relu1 = nn.Activation(activation='relu', prefix='relu1_'),
                self.conv1 = nn.Conv2D(channels=num_filter, kernel_size=(3,3), strides=stride,
                          padding=(1,1), use_bias=False, prefix='conv1_'),
                self.bn2 = nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn2_'),
                self.relu2 = nn.Activation(activation='relu', prefix='relu2_'),
                self.conv2 = nn.Conv2D(channels=num_filter, kernel_size=(3, 3), strides=(1,1),
                          padding=(1, 1), use_bias=False, prefix='conv2_')

                if not dim_match:
                    self.convsc = nn.Conv2D(channels=num_filter, kernel_size=(1,1), strides=stride, padding=(0,0),
                                            use_bias=False, prefix='sc_')

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.bottle_neck:
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))
            out = self.conv3(self.relu3(self.bn3(out)))

        else:
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))

        if not self.dim_match:
            x = self.convsc(x)
        return out + x


class ResNet(nn.HybridBlock):
    def __init__(self, units, num_stages, filter_list, num_classes,
                 image_shape, bottle_neck=True,
           bn_mom=0.9, workspace=256, memonger=False, verbose=False, **kwargs):
        """Return ResNet symbol of
        Parameters
        ----------
        units : list
            Number of units in each stage
        num_stages : int
            Number of stage
        filter_list : list
            Channel size of each stage
        num_classes : int
            Ouput size of symbol
        dataset : str
            Dataset type, only cifar10 and imagenet supports
        workspace : int
            Workspace used in convolution operator
        """
        super(ResNet, self).__init__(prefix='resnet_', **kwargs)
        self.verbose = verbose
        num_unit = len(units)
        assert (num_unit == num_stages)
        (nchannel, height, width) = image_shape

        with self.name_scope():
            stage3 = self.stage3 = nn.HybridSequential(prefix='stage3_')
            with stage3.name_scope():
                # stage 0
                stage3.add(
                    nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, gamma_initializer='ones', prefix='bn_data_')
                                )
                if height <= 32:
                    stage3.add(
                        nn.Conv2D(channels=filter_list[0], kernel_size=(3,3), strides=(1, 1), padding=(1, 1),
                            use_bias=False, prefix='conv0_')
                    )
                else:
                    stage3.add(
                        nn.Conv2D(channels=filter_list[0], kernel_size=(7,7), strides=(2,2), padding=(3,3),
                                  use_bias=False, prefix='conv0_'),
                        nn.BatchNorm(momentum=bn_mom, epsilon=2e-5, prefix='bn0_'),
                        nn.Activation(activation='relu', prefix='relu0_'),
                        nn.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding=(1,1))
                    )

                # stage 1,2,3
                for i in range(3):
                    stage3.add(
                        residual_unit(filter_list[i + 1], stride=(1 if i == 0 else 2, 1 if i == 0 else 2),
                                    dim_match=False,
                                    name='stage%d_unit%d' % (i + 1, 1),
                                    bottle_neck=bottle_neck, workspace=workspace,
                                    memonger=memonger)
                    )
                    for j in range(units[i]-1):
                        stage3.add(
                            residual_unit(filter_list[i + 1], (1, 1), dim_match=True,
                                          name='stage%d_unit%d' % (i + 1, j + 2),
                                          bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
                        )

            stage4 = self.stage4 = nn.HybridSequential(prefix='stage4_')
            with stage4.name_scope():
                stage4.add(
                    residual_unit(filter_list[4], stride=(2, 2),
                                  dim_match=False,
                                  name='stage%d_unit%d' % (4, 1),
                                  bottle_neck=bottle_neck, workspace=workspace,
                                  memonger=memonger)
                )
                for j in range(units[3] - 1):
                    stage3.add(
                        residual_unit(filter_list[4], (1, 1), dim_match=True,
                                      name='stage%d_unit%d' % (4, j + 2),
                                      bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
                    )

        self.dict = {'stage3': self.stage3,
                     'stage4': self.stage4}