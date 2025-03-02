# -*- coding: utf-8 -*-
import torch as th
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = th.relu(self.fc1(x))
        x = th.relu(self.fc2(x))
        return self.fc3(x)

    def initialize_fc_weights(self):
        for layer in (self.fc1, self.fc2, self.fc3):
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )
            th.nn.init.zeros_(layer.bias)