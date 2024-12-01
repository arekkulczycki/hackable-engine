from typing import List, Tuple, Optional

import torch as th
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        in_channels: int = 0,
        out_channels: int = 0,
        device: th.device = th.device("xpu"),
    ) -> None:
        super().__init__()

        self.reduce_block = (
            nn.Sequential(
                nn.Conv2d(
                    in_channels or channels,
                    out_channels or channels,
                    kernel_size=1,
                    stride=stride,
                ),
                # nn.BatchNorm2d(out_channels or channels),
            )
            if out_channels or in_channels
            else None
        )

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels or channels,
                out_channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(num_features=channels, device=device),
            nn.ReLU(),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=out_channels or channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
                device=device,
            ),
            nn.BatchNorm2d(num_features=out_channels or channels, device=device),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        if self.reduce_block:
            residual = self.reduce_block(residual)
        out += residual
        out = F.relu(out)
        return out


class ResidualBottleneckBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        device: th.device = th.device("xpu"),
    ) -> None:
        super().__init__()

        # First convolutional block with 1x1 kernel
        self.conv1 = self.conv2d_block(
            channels, channels // 2, 1, padding=0, act="relu", device=device
        )

        # Second convolutional block with 3x3 kernel
        self.conv2 = self.conv2d_block(
            channels // 2, channels // 2, kernel_size, padding=padding, act="relu", device=device
        )

        # Third convolutional block with 1x1 kernel and no activation function
        self.conv3 = self.conv2d_block(
            channels // 2, channels, 1, padding=0, act=None, device=device
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
            nn.Conv2d(
                in_channels, out_channels, kernel_size, bias=False, padding=padding, device=device
            ),
            nn.BatchNorm2d(out_channels, device=device),
        ]

        if act == "relu":
            layers.append(nn.ReLU())  # inplace=True))
        elif act == "lrelu":
            layers.append(nn.LeakyReLU(0.1))  # , inplace=True))

        return nn.Sequential(*layers)
