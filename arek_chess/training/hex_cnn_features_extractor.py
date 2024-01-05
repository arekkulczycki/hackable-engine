# -*- coding: utf-8 -*-

import gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HexCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    Based on NatureCNN, but using 2 convolution layers, starting with a single channel.

    Maybe a quantized `Conv2d` could be used, given discrete input, but quantizing seemed to lose input information.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        n_envs: int,
        board_size: int,
        features_dim: int,
        kernel_size: int = 5,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.n_envs = n_envs
        self.board_size = board_size

        # out = input_size + 2 * padding - dilation * (kernel - 1) - 1
        # @see https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148
        # output_channels_1 = 2 * board_size - 1

        # out = number of possible placements of the kernel on the input
        # https://arxiv.org/pdf/1603.07285v1.pdf
        # possible_placements_1 = board_size**2
        # output_1_size = 1 + board_size  # (input_size - kernel + 1) + 2 * padding
        # output_channels_2 = board_size  # = output_1_size - 1

        # possible_placements_2 = (1 + possible_placements_1 - kernel_size)**2
        # output = possible_placements_1

        # self.cnn = th.nn.Sequential(
        #     th.nn.Conv2d(1, possible_placements_1, kernel_size=board_size, padding=board_size // 2),
        #     # th.nn.BatchNorm2d(output_channels_1),
        #     th.nn.ReLU(),
        #     th.nn.Conv2d(possible_placements_1, output, kernel_size=kernel_size),
        #     # th.nn.BatchNorm2d(output_channels_2),
        #     th.nn.ReLU(),
        #     # th.nn.Conv2d(possible_placements_2, output, kernel_size=kernel_size-2, dilation=1),
        #     # th.nn.MaxPool2d(kernel_size=kernel_size-2, dilation=1),
        #     th.nn.ReLU(),
        #     th.nn.Flatten(),
        # )

        output = 16
        self.cnn = th.nn.Sequential(
            th.nn.Conv2d(1, output, kernel_size=3),  # 16 channels, 5x5 matrices
            th.nn.Tanh(),
            th.nn.Conv2d(output, output, kernel_size=2),  # 16 channels 4x4 matrices
            th.nn.Tanh(),
            th.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample().astype(np.float32)[None])).shape[1] * output

        self.linear = th.nn.Sequential(
            th.nn.Linear(n_flatten, features_dim),
            # th.nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations = th.reshape(observations, (observations.size(dim=0), 1, self.board_size, self.board_size))
        return self.cnn(observations)
        # return self.linear(self.cnn(observations))

