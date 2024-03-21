import math
from typing import NamedTuple

import gymnasium as gym
import torch as th
from torch import nn

from hackable_engine.training.network.residual_blocks import ResidualBlock


class NetworkOutputs(NamedTuple):
    pi_prob: th.Tensor
    value: th.Tensor


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: nn.Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, nn.Module)

    for module in net.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class AlphaZeroFeaturesExtractor(nn.Module):
    """Policy network for AlphaZero agent."""

    def __init__(
        self,
        observation_space: gym.Space,
        num_actions: int,
        num_res_block: int = 19,
        num_filters: int = 256,
        should_initialize_weights: bool = True
    ) -> None:
        super().__init__()
        self.features_dim = 81

        c, h, w = observation_space.shape

        conv_out_hw = calc_conv2d_output((h, w), 3, 1, 1)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        # First convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=num_filters),
            nn.ReLU(),
        )

        # Residual blocks
        res_blocks = []
        for _ in range(num_res_block):
            res_blocks.append(ResidualBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=2,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * conv_out, num_actions),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=1),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(1 * conv_out, num_fc_units),
            # nn.ReLU(),
            # nn.Linear(num_fc_units, 1),
            # nn.Tanh(),
        )

        if should_initialize_weights:
            initialize_weights(self)

    def forward(self, observations: th.Tensor) -> NetworkOutputs:
        """Given raw state x, predict the raw logits probability distribution for all actions,
        and the evaluated value, all from current player's perspective."""

        # observations = th.reshape(
        #     observations, (observations.size(dim=0), 1, 9, 9)
        # )
        # observations = th.reshape(
        #     observations, (observations.size(dim=0), 2, 9, 9)
        # )
        conv_block_out = self.conv_block(observations)
        features = self.res_blocks(conv_block_out)

        # Predict raw logits distributions wrt policy
        # pi_logits = self.policy_head(features)

        # Predict evaluated value from current player's perspective.
        return self.value_head(features)
        # return pi_logits, value
