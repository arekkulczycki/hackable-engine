# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Optional, Tuple, Type

import gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import BatchNorm2d


class HexCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    Based on NatureCNN, but using 2 convolution layers, starting with a single channel.

    Maybe a quantized `Conv2d` could be used, given discrete input, but quantizing seemed to lose input information.

    :param observation_space:
    """

    def __init__(
        self,
        observation_space: gym.Space,
        board_size: int,
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        activation_func_class: Optional[Type] = None,
    ) -> None:
        # strides = tuple(stride for _ in kernel_sizes)
        features_dim = 1152 #self._get_features_number(board_size, output_filters, kernel_sizes, strides)

        super().__init__(observation_space, features_dim)

        # out = input_size + 2 * padding - dilation * (kernel - 1) - 1
        # @see https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
        # output_channels_1 = 2 * board_size - 1

        # out = number of possible placements of the kernel on the input
        # https://arxiv.org/pdf/1603.07285v1.pdf
        # possible_placements_1 = board_size**2
        # output_1_size = 1 + board_size  # (input_size - kernel + 1) + 2 * padding
        # output_channels_2 = board_size  # = output_1_size - 1

        input_filters = [1, *output_filters[:-1]]
        layers = (
            th.nn.Conv2d(
                i_f, o_f, kernel_size=ks, stride=stride
            )
            for i_f, o_f, ks, stride in zip(input_filters, output_filters, kernel_sizes, strides)
        )
        if activation_func_class:
            layers = sum(zip(layers, (BatchNorm2d(o_f) for o_f in output_filters), cycle((activation_func_class(),))), ())
        else:
            layers = sum(zip(layers, (BatchNorm2d(o_f) for o_f in output_filters)), ())

        self.cnn = th.nn.Sequential(
            *layers,
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample())
            n_flatten = (
                self.cnn(
                    th.as_tensor(th.reshape(obs.to(th.float32), (1, 1, obs.size(dim=0), obs.size(dim=0))))
                ).shape[1]
                * output_filters[-1]
            )

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            # th.nn.ReLU(),
        )

    @staticmethod
    def _get_features_number(board_size, output_filters, kernel_sizes, strides):
        # for of, ks, s in zip(output_filters, kernel_sizes, strides):
        #     output_size = (size - ks) // s + 1
        #     filters = of * output_size
        return output_filters[0] * ((board_size - kernel_sizes[0]) // strides[0] + 1)**2

    def forward(self, observations: th.Tensor) -> th.Tensor:
        board_size = observations.size(dim=1)
        observations = th.reshape(
            observations, (observations.size(dim=0), 1, board_size, board_size)
        )
        return self.cnn(observations)
        # return self.linear(self.cnn(observations))
