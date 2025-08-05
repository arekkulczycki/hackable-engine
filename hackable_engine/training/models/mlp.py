import torch as th
from torch import nn

from hackable_engine.training.models import BaseModule


class MLP(BaseModule):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.mlp = (self.fc1, self.fc2, self.fc3)

        # self.initialize_fc_weights()

    def extract_features(self, x: th.Tensor):
        return x

    def make_decision(self, x: th.Tensor):
        for layer in self.mlp[:-1]:
            x = th.relu(layer(x))
        return self.mlp[-1](x)

    def initialize_fc_weights(self):
        for layer in (self.fc1, self.fc2, self.fc3):
            # th.nn.init.kaiming_uniform_(layer.weight)
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            # th.nn.init.xavier_normal_(layer.weight)
            th.nn.init.zeros_(layer.bias)
