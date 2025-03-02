# -*- coding: utf-8 -*-
from typing import Tuple

import gymnasium as gym
import torch as th
from nptyping import NDArray
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from hackable_engine.training.hyperparams import LEARNING_RATE
from hackable_engine.training.network.residual_blocks import ResidualBlock


class HexResnetRawFeaturesExtractor(BaseFeaturesExtractor):
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
        resnet_layers: Tuple[int, ...] = (1,),
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        should_normalize: bool = True,
        should_initialize_weights: bool = True,
    ) -> None:
        device = th.device("xpu")
        obs = th.as_tensor(observation_space.sample()).to(th.float32)

        features_dim = board_size**2 * 2

        super().__init__(observation_space, features_dim, learning_rate)

        # input_filters = [obs.size(dim=0), *output_filters[:-1]]
        # self.resnet = []
        # for n_layers, in_f, out_f, kernel in zip(
        #     resnet_layers,
        #     input_filters,
        #     output_filters,
        #     kernel_sizes,
        # ):
        #     conv_layers = (
        #         th.nn.Conv2d(
        #             i_f, o_f, kernel_size=ks, padding=(ks - 1) // 2, device=device
        #         )
        #     )
        #
        #     self.resnet.extend(self._get_resnet(
        #         n_layers,
        #     ))
        self.resnet = [
            ResidualBlock(128, 5, 1, 2, in_channels=2, device=device),
            ResidualBlock(128, 5, 1, 2, device=device),
            ResidualBlock(128, 5, 1, 2, out_channels=2, device=device),
        ]

        self.network = th.nn.Sequential(
            *self.resnet,
            th.nn.Flatten(),
        )

        if should_initialize_weights:
            self._initialize_weights()

    @staticmethod
    def _get_resnet(
        num_layers: int,
        num_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        device: th.device,
    ):
        return [
            ResidualBlock(num_channels, kernel_size, stride, padding, device=device)
            for _ in range(num_layers)
        ]

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

    def forward(self, observations: NDArray) -> th.Tensor:  # th.Tensor) -> th.Tensor:
        return self.network(observations)
