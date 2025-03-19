# -*- coding: utf-8 -*-
from abc import ABC

import torch as th

from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class CriticMixin(BaseModule, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mlp[0].__init__(
            self.mlp[0].in_features + 1, self.mlp[0].out_features, device=Device.XPU
        )

    def forward(self, x: th.Tensor, a: th.Tensor):
        x = x.flatten(
            1, -1
        )  # will have no effect on 2-dim tensors, will flatten 3-dim into 2-dim
        x = self.extract_features(x)

        x = th.cat([x, a.unsqueeze(-1)], -1)
        x = self.make_decision(x)

        return x.flatten(1, -1)
