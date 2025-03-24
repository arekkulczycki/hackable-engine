# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SGConv

from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class GraphSG(BaseModule):

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
        is_seq = False,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.gnn_shape = gnn_shape
        self.mlp_shape = mlp_shape
        self.is_seq = is_seq

        self.setup_graph_feature_extractor(edge_index, gnn_shape)
        self.setup_mlp(gnn_shape, mlp_shape, output_size, is_seq)

        self.initialize_gnn_weights()
        self.initialize_mlp_weights()

    def setup_graph_feature_extractor(self, edge_index, shape):
        self.edge_index = edge_index.to(Device.XPU)

        self.res_proj_0 = nn.Linear(self.node_features, shape[0], device=Device.XPU)
        self.res_proj_1 = nn.Linear(shape[0], shape[1], device=Device.XPU)
        self.res_proj_2 = nn.Linear(shape[1], shape[2], device=Device.XPU)
        self.res_proj_3 = nn.Linear(shape[2], shape[3], device=Device.XPU)
        self.res_proj_4 = nn.Linear(shape[3], shape[4], device=Device.XPU)
        self.conv1 = SGConv(self.node_features, shape[0], K=1)
        self.conv2 = SGConv(shape[0], shape[1], K=1)
        self.conv3 = SGConv(shape[1], shape[2], K=1)
        self.conv4 = SGConv(shape[2], shape[3], K=1)
        self.conv5 = SGConv(shape[3], shape[4], K=1)
        # self.conv1 = GCNConv(input_size, shape[0])
        # self.conv2 = GCNConv(shape[0], shape[1])
        # self.conv3 = GCNConv(shape[1], shape[2])
        # self.conv4 = GCNConv(shape[2], shape[3])
        self.gnn = (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        # self.norm1 = GraphNorm(shape[0])
        # self.norm2 = GraphNorm(shape[1])
        # self.norm3 = GraphNorm(shape[2])

    def setup_mlp(self, gnn_shape, mlp_shape, output_size, is_seq = False):
        mlp = []
        prev_size = gnn_shape[-1] * self.node_count if is_seq else gnn_shape[-1]
        for size in [*mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=Device.XPU))
            prev_size = size

        self.mlp = tuple(mlp)

    def forward(self, x, *args):
        # x = x.flatten(1, -1)  # will have no effect on 2-dim tensors, will flatten 3-dim into 2-dim
        x = self.extract_features(x)
        if self.is_seq:
            x = x.flatten(-2, -1)
        x = self.make_decision(x)
        return x.flatten(1, -1)

    def extract_features(self, x):
        res0 = self.res_proj_0(x)
        x = F.relu(self.conv1(x, self.edge_index))
        # x = self.norm1(x, batch, batch_size)
        x = x + res0

        res1 = self.res_proj_1(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = F.relu(self.conv2(x, self.edge_index))
        # x = self.norm2(x, batch, batch_size)
        x = x + res1

        res2 = self.res_proj_2(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = F.relu(self.conv3(x, self.edge_index))
        # x = self.norm3(x, batch, batch_size)
        x = x + res2

        res3 = self.res_proj_3(x)
        x = F.relu(self.conv4(x, self.edge_index))
        x = x + res3

        res4 = self.res_proj_4(x)
        x = F.relu(self.conv5(x, self.edge_index))
        x = x + res4

        return x

    def initialize_gnn_weights(self):
        for layer in self.gnn:
            th.nn.init.kaiming_uniform_(
                layer.lin.weight, mode="fan_in", nonlinearity="relu"
            )
            if layer.lin.bias is not None:
                th.nn.init.zeros_(layer.lin.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="tanh")
            th.nn.init.zeros_(layer.bias)

    # def make_decision(self, x: th.Tensor):
    #     for layer in self.mlp[:-1]:
    #         x = th.tanh(layer(x))
    #     return self.mlp[-1](x)
