"""@package docstring
module containing basic building blocks, like our base Conv+Norm+Act class etc.
"""
import torch
import torch.nn as nn


class ConvBatchnormAct(nn.Module):
    """Simple building block that unifies a convolution, activation and batch normalization
        args:
            in_channels(int): number of input features
            out_channels(int): number of output features
            kernel_size(int): kernel size of convolution, default 3
            stride(int): stride of convolution, default 1
            dilation(int): dilation (spacing) of kernel elements, default 1 (dense conv kernels)
            no_activation(bool): if true, does not apply batch normalization and activation function
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 act,
                 kernel_size: tuple[int, int] | int = 3,
                 stride: tuple[int, int] | int = 1,
                 dilation = 1,
                 no_activation = False,
                 separable = False,
                 groups = 1,
                 act_inplace = True,
                 bias = True):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = (kernel_size // 2)*dilation
        else:
            padding = ((kernel_size[0] // 2)*dilation, (kernel_size[1] // 2)*dilation)
        if not no_activation:
            if separable:
                self.map = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups = in_channels, bias = bias),
                                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, bias = False),
                                            nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-4), act(inplace=act_inplace))
            else:
                self.map = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = False, groups = groups),
                                         nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-4),
                                         act(inplace=act_inplace))
        else:
            if separable:
                self.map = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups = in_channels, bias = bias),
                                            nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, bias = bias))
            else:
                self.map = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias = bias, groups=groups)
        self._init_weight()
        self.out_channels = out_channels
    def forward(self, x):
        return self.map(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, a=1e-2)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class ResidualBlock(nn.Module):
    """TODO: Docstring"""
    def __init__(self, in_channels, channels, act, depth = 5, downsample = True, separable=False, stride = 1):
        super().__init__()
        input_stride = 2 if downsample else 1
        if stride != 1:
            input_stride = stride
        self.conv_in_reduce = ConvBatchnormAct(in_channels, channels, act, stride=input_stride, separable=separable)
        layers = []
        for _ in range(depth):
            layers.append(ResidualModule(channels, act = act, separable=separable))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_in_reduce(x)
        x = self.layers(x)
        return x

class ResidualModule(nn.Module):
    """TODO: Docstring"""
    def __init__(self, in_channels, act, separable = False):
        super(ResidualModule, self).__init__()
        self.conv_1 = ConvBatchnormAct(in_channels, in_channels, act, separable=separable)
        self.conv_2 = ConvBatchnormAct(in_channels, in_channels, act, no_activation=True, separable=separable)
        self.out = nn.Sequential(nn.BatchNorm2d(in_channels, eps = 1e-4, momentum=0.01), act())

    def forward(self, x):
        y = self.conv_1(x)
        y = self.conv_2(y)
        return self.out(x + y)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1, 0.02)
                m.bias.data.zero_()


class DenseUpsamplingConvolution(nn.Module):
    """ This block provides DeepLab-Sytle dense upsampling convolution
        args:
            in_channels:                number of input channels
            out_channels:               output_channels
            upsample_factor:            scaling factor
    """
    def __init__(self, in_channels, out_channels, upsample_factor, no_activation = False):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, (upsample_factor**2) * out_channels, kernel_size=3, padding = 1) if no_activation else ConvBatchnormAct(in_channels, upsample_factor**2 * out_channels, act=nn.LeakyReLU, kernel_size=3)
        self.shuffle = nn.PixelShuffle(upsample_factor)
    def forward(self, x):
        x = self.conv_in(x)
        x = self.shuffle(x)
        return x