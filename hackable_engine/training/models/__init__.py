from abc import ABC, abstractmethod
from typing import Any

import gymnasium as gym
import torch as th
from torch import nn


class BaseModule(nn.Module, ABC):
    env: gym.vector.vector_env.VectorEnv
    gnn_shape: tuple[int, ...]
    mlp_shape: tuple[int, ...]
    residuals: nn.ModuleList
    norms: nn.ModuleList
    gnn: nn.ModuleList
    mlp: nn.ModuleList
    control_mlp: nn.ModuleList
    value_head: nn.Sequential
    policy_head: nn.Sequential

    def forward(self, x: th.Tensor, *args: Any):
        x = self.extract_features(x)
        return self.make_decision(x)

    @abstractmethod
    def extract_features(self, x: th.Tensor):
        ...

    def make_decision(self, x: th.Tensor):
        for layer in self.mlp[:-1]:
            x = th.relu(layer(x))
        return self.mlp[-1](x)
