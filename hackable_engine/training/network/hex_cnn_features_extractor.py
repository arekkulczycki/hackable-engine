# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Optional, Tuple, Type

import gym
import torch as th
from nptyping import NDArray
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import BatchNorm2d

from hackable_engine.training.hyperparams import LEARNING_RATE
from hackable_engine.training.network.residual_blocks import (
    ResidualBlock,
    ResidualBottleneckBlock,
)


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
        learning_rate: float = LEARNING_RATE,
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        resnet_layers: int = 0,
        resnet_channels: int = 1,
        should_normalize: bool = True,
        activation_fn: Optional[Type] = None,
        should_initialize_weights: bool = True,
    ) -> None:
        device = th.device("xpu")
        obs = th.as_tensor(observation_space.sample()).to(th.float32)
        if output_filters:
            features_dim = self._get_features_number(
                board_size, output_filters, kernel_sizes, strides
            )
        else:
            features_dim = obs.size(dim=1) ** 2

        super().__init__(observation_space, features_dim, learning_rate)

        n_channels = obs.size(dim=0)
        input_filters = [n_channels, *output_filters[:-1]]
        conv_layers = (
            th.nn.Conv2d(i_f, o_f, kernel_size=ks, stride=stride, device=device)
            for i_f, o_f, ks, stride in zip(
                input_filters, output_filters, kernel_sizes, strides
            )
        )
        layer_generators = [conv_layers]
        if should_normalize:
            layer_generators.append(
                (BatchNorm2d(o_f, device=device) for o_f in output_filters)
            )
        if activation_fn:
            layer_generators.append(cycle((activation_fn(),)))

        layers = sum(zip(*layer_generators), ())
        resnet = self._get_resnet(
            resnet_layers,
            output_filters[-1] if output_filters else resnet_channels,
            device,
        )

        self.network = th.nn.Sequential(
            *layers,
            *resnet,
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        # with th.no_grad():
        #     obs = th.reshape(obs, (1, n_channels, board_size, board_size))
        #     n_flatten = (
        #         self.network(obs).shape[1] * output_filters[-1]
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
    def _get_resnet(num_layers: int, num_channels: int, device: th.device):
        return [ResidualBlock(num_channels, device=device) for _ in range(num_layers)]
        # return [ResidualBottleneckBlock(num_channels, device=device) for _ in range(num_layers)]

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
        n_modules = len(modules)
        for i in range(n_modules):
            module = modules[i]
            if isinstance(module, th.nn.Conv2d):
                if (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.ReLU)) or (
                    n_modules > i + 1 and isinstance(modules[i + 1], th.nn.ReLU)
                ):
                    th.nn.init.kaiming_normal_(
                        module.weight, mode="fan_out", nonlinearity="relu"
                    )
                else:
                    th.nn.init.xavier_normal_(module.weight)

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)
            elif isinstance(module, th.nn.Linear):
                th.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)
            elif isinstance(module, th.nn.BatchNorm2d):
                th.nn.init.constant_(module.weight, 1)
                th.nn.init.constant_(module.bias, 0)

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
