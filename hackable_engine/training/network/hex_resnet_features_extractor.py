# -*- coding: utf-8 -*-
from typing import Tuple

import gym
import torch as th
from nptyping import NDArray
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import BatchNorm2d

from hackable_engine.training.hyperparams import LEARNING_RATE
from hackable_engine.training.network.residual_blocks import (
    ResidualCustomBlock,
    ResidualBlock,
)


class HexResnetFeaturesExtractor(BaseFeaturesExtractor):
    """
    Simple CNN constructed by a number of alternating `Conv2d` and `BatchNorm2d` layers, depending on given arguments.

    @see https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

    output to the next layer is a map of possible placements of the kernel on the input
    https://arxiv.org/pdf/1603.07285v1.pdf
    """

    def __init__(
        self,
        observation_space: gym.Space,
        board_size: int,
        learning_rate: float = LEARNING_RATE,
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        resnet_layers: int = 1,
        should_normalize: bool = True,
        should_initialize_weights: bool = True,
    ) -> None:
        device = th.device("xpu")
        obs = th.as_tensor(observation_space.sample()).to(th.float32)
        if output_filters:
            features_dim = self._get_features_number(board_size, output_filters, kernel_sizes, strides)
        else:
            features_dim = obs.size(dim=1) ** 2
        # features_dim = 1152
        features_dim = output_filters[-1]

        super().__init__(observation_space, features_dim, learning_rate)

        input_filters = [obs.size(dim=0), *output_filters[:-1]]

        self.resnet = []
        for i in range(resnet_layers):
            conv_layers = (
                th.nn.Conv2d(
                    i_f, o_f, kernel_size=ks, padding=(ks - 1) // 2, device=device
                )
                for i_f, o_f, ks in zip(
                    input_filters if i == 0 else output_filters,
                    output_filters,
                    kernel_sizes,
                )
            )

            if should_normalize:
                convolutions = list(
                    zip(
                        conv_layers,
                        (BatchNorm2d(o_f, device=device) for o_f in output_filters),
                    )
                )
            else:
                convolutions = [(layer,) for layer in conv_layers]

            self.resnet.append(ResidualCustomBlock(convolutions))

        self.network = th.nn.Sequential(
            *self.resnet,
            # th.nn.MaxPool2d(kernel_size=3, stride=3),
            # *self._get_resnet(resnet_layers, output_filters[-1], device),
            th.nn.Conv2d(output_filters[-1], output_filters[-1], kernel_size=board_size, device=device),
            # *[
            #     th.nn.Conv2d(i_f, o_f, kernel_size=ks, stride=stride, device=device)
            #     for i_f, o_f, ks, stride in zip(
            #         input_filters if i == 0 else output_filters,
            #         output_filters,
            #         kernel_sizes,
            #         strides,
            #     )
            # ],
            # *self._get_resnet(resnet_layers, output_filters[-1], device=device),
            th.nn.Flatten(),
        )

        if should_initialize_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights for layers:
            `Conv2d`: Xavier if using Tanh activation function
            `Linear`: Kaiming which accounts for non-linear activation function like ReLU
        @see https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
        @see https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
        @see https://alexkelly.world/posts/initialization_neural_networks/
        """

        modules = list(self.network.modules())
        for i in range(len(modules)):
            module = modules[i]
            if isinstance(module, th.nn.Conv2d):
                th.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)
            elif isinstance(module, th.nn.BatchNorm2d):
                th.nn.init.constant_(module.weight, 1)
                th.nn.init.constant_(module.bias, 0)

    @staticmethod
    def _get_resnet(num_layers: int, num_channels: int, device: th.device):
        return [ResidualBlock(num_channels, device=device) for _ in range(num_layers)]
        # return [ResidualBottleneckBlock(num_channels, device=device) for _ in range(num_layers)]

    @staticmethod
    def _get_features_number(board_size, output_filters, kernel_sizes, strides):
        """Input to the last layer"""

        limit = len(strides) - 1
        filters = 0

        i = 0
        for of, ks, s in zip(output_filters, kernel_sizes, strides):
            board_size = (board_size - ks) // s + 1
            if i == limit:
                filters = of * board_size**2
                break
            else:
                i += 1

        return filters

    def forward(self, observations: NDArray) -> th.Tensor:  # th.Tensor) -> th.Tensor:
        return self.network(observations)
