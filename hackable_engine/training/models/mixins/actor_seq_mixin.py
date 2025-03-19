# -*- coding: utf-8 -*-
from abc import ABC

import torch as th
from torch import nn

from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class ActorMixin(BaseModule, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fc_mean = nn.Linear(self.mlp_shape[-1], 1, device=Device.XPU)
        self.fc_logstd = nn.Linear(self.mlp_shape[-1], 1, device=Device.XPU)
        self.register_buffer(
            "action_scale",
            th.tensor(
                1.0,
                dtype=th.float32,
                device=Device.XPU,
            ),
        )
        self.register_buffer(
            "action_bias",
            th.tensor(
                0.0,
                dtype=th.float32,
                device=Device.XPU,
            ),
        )

    def make_decision(self, x: th.Tensor):
        for layer in self.mlp[:-1]:
            x = th.relu(layer(x))

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )

        return mean.flatten(1, -1), log_std.flatten(1, -1)

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh(mean) * self.action_scale + self.action_bias

        # TODO: this is for LogNormal, erase for Normal distribution
        # action = action * 2 - 1

        return action, log_prob, mean, std
