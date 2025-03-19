# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class GraphGAT(BaseModule):

    def __init__(
        self,
        node_count,
        node_features,
        output_size,
        batch_size,
        num_envs,
        gnn_shape,
        mlp_shape,
        edge_index,
        edge_types,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.gnn_shape = gnn_shape
        self.mlp_shape = mlp_shape

        self.edge_index = edge_index.to(Device.XPU)
        self.edge_types = edge_types.to(Device.XPU)

        self.setup_graph_feature_extractor()
        self.setup_mlp(output_size)

        # self.initialize_gnn_weights()
        self.initialize_mlp_weights()

    def setup_graph_feature_extractor(self):
        heads = 4
        self.res_proj_0 = nn.Linear(self.node_features, self.gnn_shape[0] * heads, device=Device.XPU)
        self.res_proj_1 = nn.Linear(self.gnn_shape[0] * heads, self.gnn_shape[1] * heads, device=Device.XPU)
        self.res_proj_2 = nn.Linear(self.gnn_shape[1] * heads, self.gnn_shape[2] * heads, device=Device.XPU)
        self.res_proj_3 = nn.Linear(self.gnn_shape[2] * heads, self.gnn_shape[3], device=Device.XPU)
        self.conv1 = GATv2Conv(self.node_features, self.gnn_shape[0], heads=heads, concat=True, edge_dim=3)
        self.conv2 = GATv2Conv(self.gnn_shape[0] * heads, self.gnn_shape[1], heads=heads, concat=True, edge_dim=3)
        self.conv3 = GATv2Conv(self.gnn_shape[1] * heads, self.gnn_shape[2], heads=heads, concat=True, edge_dim=3)
        self.conv4 = GATv2Conv(self.gnn_shape[2] * heads, self.gnn_shape[3], heads=1, concat=True, edge_dim=3)
        self.gnn = (self.conv1, self.conv2, self.conv3, self.conv4)
        # self.norm1 = GraphNorm(shape[0])
        # self.norm2 = GraphNorm(shape[1])
        # self.norm3 = GraphNorm(shape[2])

    def setup_mlp(self, output_size):
        mlp = []
        prev_size = self.gnn_shape[-1]
        for size in [*self.mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=Device.XPU))
            prev_size = size

        self.mlp = tuple(mlp)

    def forward(self, x, *args):
        # x = x.flatten(1, -1)  # will have no effect on 2-dim tensors, will flatten 3-dim into 2-dim
        x = self.extract_features(x)
        x = self.make_decision(x)
        return x.flatten(1, -1)

    def extract_features(self, x):
        # expecting x to be already one-hot
        # x = F.one_hot((x + 1).long(), num_classes=3).to(TH_FLOAT_TYPE)

        # GATConv expects a large graph instead of batches, so we'll rely on edge_index to unfold the graph later
        batch_size = x.shape[0]
        x = x.view(-1, self.node_features)

        res0 = self.res_proj_0(x)
        x = F.relu(self.conv1(x, self.edge_index, edge_attr=self.edge_types))
        # x = self.norm1(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res0
        res1 = self.res_proj_1(x)
        x = F.relu(self.conv2(x, self.edge_index, edge_attr=self.edge_types))
        # x = self.norm2(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res1
        res2 = self.res_proj_2(x)
        x = F.relu(self.conv3(x, self.edge_index, edge_attr=self.edge_types))
        # x = self.norm3(x, batch, batch_size)
        x = x + res2
        res3 = self.res_proj_3(x)
        x = F.relu(self.conv4(x, self.edge_index, edge_attr=self.edge_types))
        x = x + res3

        # unfold the batched graph
        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        for layer in self.gnn:
            th.nn.init.kaiming_uniform_(
                layer.lin.weight, mode="fan_in", nonlinearity="relu"
            )
            if layer.lin.bias is not None:
                th.nn.init.zeros_(layer.lin.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp:
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )
            th.nn.init.zeros_(layer.bias)
