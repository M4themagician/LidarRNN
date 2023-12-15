import torch.nn as nn
from network.building_blocks import ResidualBlock, DenseUpsamplingConvolution

class LidarRNN(nn.Module):
    down_channels = [64, 128, 256, 512]
    up_channels = [512, 256]
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for c in self.down_channels:
            layers.append(ResidualBlock(in_channels, c, nn.ReLU, depth = 2, separable=True))
            in_channels = c
        
        for c in self.up_channels:
            layers.append(DenseUpsamplingConvolution(in_channels, c, 2))
            in_channels = c
        out_conv = nn.Sequential(ResidualBlock(in_channels, 64, nn.ReLU, depth=2, downsample=False, separable=True), nn.Conv2d(64, 2+9, kernel_size=1))
        self.layers = nn.Sequential(*layers, out_conv)

    def forward(self, x):
        return self.layers(x)