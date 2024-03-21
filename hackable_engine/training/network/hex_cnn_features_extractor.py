# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Optional, Tuple, Type

import gym
import torch as th
from nptyping import NDArray
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import BatchNorm2d

from hackable_engine.training.network.residual_blocks import ResidualBlock


class HexCnnFeaturesExtractor(BaseFeaturesExtractor):
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
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        resnet_layers: int = 0,
        should_normalize: bool = True,
        activation_fn: Optional[Type] = None,
        should_initialize_weights: bool = True,
    ) -> None:
        features_dim = self._get_features_number(board_size, output_filters, kernel_sizes, strides)

        super().__init__(observation_space, features_dim)

        obs = th.as_tensor(observation_space.sample()).to(th.float32)
        n_channels = obs.size(dim=0)

        input_filters = [n_channels, *output_filters[:-1]]
        conv_layers = (
            th.nn.Conv2d(
                i_f, o_f, kernel_size=ks, stride=stride
            )
            for i_f, o_f, ks, stride in zip(input_filters, output_filters, kernel_sizes, strides)
        )
        layer_generators = [
            conv_layers
        ]
        if should_normalize:
            layer_generators.append((BatchNorm2d(o_f) for o_f in output_filters))
        if activation_fn:
            layer_generators.append(cycle((activation_fn(),)))

        layers = sum(zip(*layer_generators), ())
        resnet = self._get_resnet(resnet_layers, output_filters[-1])

        self.cnn = th.nn.Sequential(
            # *layers[:-1],  # used if the last conv layer returns 1x1 and cannot be normalized
            *layers,
            *resnet,
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     obs = th.reshape(obs, (1, n_channels, board_size, board_size))
        #     n_flatten = (
        #         self.cnn(obs).shape[1] * output_filters[-1]
        #     )
        #
        # self.linear = th.nn.Sequential(
        #     th.nn.Linear(n_flatten, features_dim),
        #     th.nn.Sigmoid(),
        #     # th.nn.ReLU(),
        # )

        if should_initialize_weights:
            self._initialize_weights()

    @staticmethod
    def _get_resnet(num_layers: int, num_channels: int):
        return [ResidualBlock(num_channels) for _ in range(num_layers)]

    def _initialize_weights(self) -> None:
        """
        Initialize weights for layers:
            `Conv2d`: Xavier because of using Tanh activation function
            `Linear`: Kaiming which accounts for non-linear activation function like ReLU
        @see https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
        @see https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
        @see https://alexkelly.world/posts/initialization_neural_networks/
        """

        for module in self.modules():
            if isinstance(module, th.nn.Conv2d):
                th.nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)
            elif isinstance(module, th.nn.Linear):
                th.nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)

    @staticmethod
    def _get_features_number(board_size, output_filters, kernel_sizes, strides):
        """Input to the last layer"""

        limit = len(strides) - 1
        filters = 0

        i = 0
        for of, ks, s in zip(output_filters, kernel_sizes, strides):
            board_size = (board_size - ks) // s + 1
            if i == limit:
                filters = of * board_size ** 2
                break
            else:
                i += 1

        # raise ValueError(filters)  # just to check the number of filters with a crash
        return filters

    def forward(self, observations: NDArray) -> th.Tensor:  # th.Tensor) -> th.Tensor:
        # observations = observations.reshape((observations.shape[0], 1, 9, 9))
        return self.cnn(observations)
