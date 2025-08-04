# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv
from torch_geometric.nn.norm import GraphNorm, LayerNorm

from hackable_engine.training.utils.device import Device
from hackable_engine.training.models import BaseModule


class GraphGIN(BaseModule):

    def __init__(
        self,
        node_count,
        node_features,
        output_size,
        batch_size,
        dropouts,
        num_envs,
        gnn_shape,
        mlp_shape,
        edge_index,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.dropouts = dropouts
        self.gnn_shape = gnn_shape
        self.mlp_shape = mlp_shape

        self.edge_index = edge_index.to(Device.XPU)
        self.batch_edge_index = self.get_batch_edge_index(node_count, batch_size)

        self.setup_graph_feature_extractor()
        self.setup_mlp(gnn_shape, mlp_shape, output_size)

        self.initialize_gnn_weights()
        self.initialize_residuals_weights()
        self.initialize_mlp_weights()

    def get_batch_edge_index(self, node_count, batch_size):
        batch_edge_index = []
        for i in range(batch_size):
            batch_edge_index.append(self.edge_index + i * node_count)
        return th.cat(batch_edge_index, dim=1).to(Device.XPU)

    def setup_graph_feature_extractor(self):
        self.input_proj = nn.Linear(
            self.node_features, self.gnn_shape[0], device=Device.XPU
        )

        convs = []
        residuals = []
        for in_channels, out_channels in zip(self.gnn_shape[:-1], self.gnn_shape[1:]):
            if in_channels == out_channels:
                residuals.append(nn.Identity())
            else:
                residuals.append(
                    nn.Linear(in_channels, out_channels, device=Device.XPU)
                )

            mlp = nn.Sequential(
                nn.Linear(in_channels, out_channels, device=Device.XPU),
                LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(self.dropouts),
                nn.Linear(out_channels, out_channels, device=Device.XPU),
            )
            convs.append(GINConv(mlp, train_eps=True))

        self.residuals = nn.ModuleList(residuals)
        self.gnn = nn.ModuleList(convs)

    def setup_mlp(self, gnn_shape, mlp_shape, output_size):
        mlp = []
        control_mlp = []

        prev_size = gnn_shape[-1]
        for size in [*mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=Device.XPU))
            prev_size = size

        prev_size = gnn_shape[-1]
        for size in [*mlp_shape, 3]:
            control_mlp.append(nn.Linear(prev_size, size, device=Device.XPU))
            prev_size = size

        self.mlp = nn.ModuleList(mlp)
        self.control_mlp = nn.ModuleList(control_mlp)

    def forward(self, x, *args):
        x = x.flatten(0, 1)
        x = self.extract_features(x)

        if self.training:
            x, control = self.make_decision(x)
            return x.flatten(1, -1), control
        else:
            # return self.make_decision(x).flatten()
            return self.make_decision(x).flatten(1, -1)

    def extract_features(self, x):
        if x.shape[0] == self.batch_size:
            batch_size = self.batch_size
            edge_index = self.batch_edge_index
        else:
            batch_size = x.shape[0]
            edge_index = self.get_batch_edge_index(self.node_count, batch_size)

        x = x.view(-1, self.node_features)

        x = self.input_proj(x)
        for conv, residual in zip(self.gnn, self.residuals):
            h = conv(x, edge_index)
            # x = F.dropout(F.relu(h + residual(x)), p=self.dropouts, training=self.training)
            # x = F.relu(h + residual(x))
            x = F.dropout(F.relu(h), p=self.dropouts, training=self.training) + residual(x)

        # unfold the batched graph
        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        gin: GINConv
        for gin in self.gnn:
            for layer in gin.nn:
                if isinstance(layer, nn.Linear):
                    th.nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        th.nn.init.zeros_(layer.bias)

    def initialize_residuals_weights(self):
        for layer in self.residuals:
            if layer.__class__.__name__ == "Identity":
                continue
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            th.nn.init.zeros_(layer.bias)

    def initialize_mlp_weights(self):
        for layer in list(self.mlp) + list(self.control_mlp):
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            )
            # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="tanh")
            th.nn.init.zeros_(layer.bias)

    def make_decision(self, x: th.Tensor):
        mlp_x = x
        control_x = x
        for layer in self.mlp[:-1]:
            mlp_x = F.leaky_relu(layer(mlp_x), negative_slope=0.05)
            # mlp_x = F.dropout(F.leaky_relu(layer(x)), p=0.1, training=self.training)

        if self.training:
            for layer in self.control_mlp[:-1]:
                control_x = F.leaky_relu(layer(control_x), negative_slope=0.05)

            return self.mlp[-1](mlp_x), self.control_mlp[-1](control_x)
        else:
            return self.mlp[-1](mlp_x)
