from typing import List, Tuple

import torch as th
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):

    def __init__(self, channels: int, device: th.device = th.device("xpu")) -> None:
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(num_features=channels, device=device),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(num_features=channels, device=device),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class ResidualBottleneckBlock(nn.Module):
    def __init__(self, channels, device: th.device = th.device("xpu")):
        super().__init__()

        # First convolutional block with 1x1 kernel
        self.conv1 = self.conv2d_block(
            channels, channels//2, 1, padding=0, act='relu', device=device
        )

        # Second convolutional block with 3x3 kernel
        self.conv2 = self.conv2d_block(
            channels//2, channels//2, 3, padding=1, act='relu', device=device
        )

        # Third convolutional block with 1x1 kernel and no activation function
        self.conv3 = self.conv2d_block(
            channels//2, channels, 1, padding=0, act=None, device=device
        )

    def forward(self, x):
        # Pass the input through the convolutional blocks
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)
        x_out = self.conv3(x_out)

        # Add the input to the output of the convolutional blocks
        return x + x_out

    @staticmethod
    def conv2d_block(in_channels, out_channels, kernel_size, padding, act, device):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, device=device),
            nn.BatchNorm2d(out_channels, device=device),
        ]

        if act == 'relu':
            layers.append(nn.ReLU())  #inplace=True))
        elif act == 'lrelu':
            layers.append(nn.LeakyReLU(0.1))  #, inplace=True))

        return nn.Sequential(*layers)


class ResidualCustomBlock(nn.Module):

    def __init__(self, convolutions: List[Tuple[nn.Module]]) -> None:
        super().__init__()

        self.conv_blocks = []
        number_of_convolutions = len(convolutions)

        for i, convolution_modules in enumerate(convolutions):
            if i == number_of_convolutions - 1:
                self.conv_blocks.append(nn.Sequential(*convolution_modules))
            else:
                self.conv_blocks.append(nn.Sequential(*convolution_modules, nn.ReLU()))

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x

        for block in self.conv_blocks:
            out = block(x)

        out += residual
        out = F.relu(out)
        return out
