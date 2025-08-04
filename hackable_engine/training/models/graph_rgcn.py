# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import FastRGCNConv, RGCNConv

from hackable_engine.training.utils.device import Device
from hackable_engine.training.models import BaseModule


class GraphRGCN(BaseModule):

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
        edge_types,
        device=Device.XPU,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.dropouts = dropouts
        self.gnn_shape = gnn_shape
        self.mlp_shape = mlp_shape
        self.device = device
        self.use_res = True

        self.edge_index = edge_index.to(device)
        self.batch_edge_index = self.get_batch_edge_index(batch_size)
        self.edge_types = edge_types.to(device)
        self.batch_edge_types = self.get_batch_edge_types(batch_size)

        self.setup_graph_feature_extractor()
        self.setup_mlp(gnn_shape, mlp_shape, output_size)

        # self.initialize_gnn_weights()
        self.initialize_mlp_weights()

    def get_batch_edge_index(self, batch_size):
        batch_edge_index = []
        for i in range(batch_size):
            batch_edge_index.append(self.edge_index + i * self.node_count)
        return th.cat(batch_edge_index, dim=1).to(self.device)

    def get_batch_edge_types(self, batch_size):
        batch_edge_types = []
        for i in range(batch_size):
            batch_edge_types.append(self.edge_types)
        return th.cat(batch_edge_types, dim=0).to(self.device).to(th.long)

    def setup_graph_feature_extractor(self):
        if self.use_res:
            self.res_proj_0 = nn.Linear(self.node_features, self.gnn_shape[0], device=self.device)
            self.res_proj_1 = nn.Linear(self.gnn_shape[0], self.gnn_shape[1], device=self.device)
            self.res_proj_2 = nn.Linear(self.gnn_shape[1], self.gnn_shape[2], device=self.device)
            self.res_proj_3 = nn.Linear(self.gnn_shape[2], self.gnn_shape[3], device=self.device)
            self.res_proj_4 = nn.Linear(self.gnn_shape[3], self.gnn_shape[4], device=self.device)
            # self.res_proj_5 = nn.Linear(
            #     self.gnn_shape[4], self.gnn_shape[5], device=self.device
            # )
        self.conv1 = RGCNConv(
            self.node_features,
            self.gnn_shape[0],
            num_relations=3,
        )
        self.conv2 = RGCNConv(
            self.gnn_shape[0],
            self.gnn_shape[1],
            num_relations=3,
        )
        self.conv3 = RGCNConv(
            self.gnn_shape[1],
            self.gnn_shape[2],
            num_relations=3,
        )
        self.conv4 = RGCNConv(
            self.gnn_shape[2],
            self.gnn_shape[3],
            num_relations=3,
        )
        self.conv5 = RGCNConv(
            self.gnn_shape[3],
            self.gnn_shape[4],
            num_relations=3,
        )
        # self.conv6 = RGCNConv(
        #     self.gnn_shape[4],
        #     self.gnn_shape[5],
        #     num_relations=3,
        # )
        self.gnn = (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)  # , self.conv6)

    def setup_mlp(self, gnn_shape, mlp_shape, output_size):
        mlp = []
        control_mlp = []

        prev_size = gnn_shape[-1]
        for size in [*mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=self.device))
            prev_size = size

        prev_size = gnn_shape[-1]
        for size in [*mlp_shape, 3]:
            control_mlp.append(nn.Linear(prev_size, size, device=self.device))
            prev_size = size

        self.mlp = nn.ModuleList(mlp)
        self.control_mlp = nn.ModuleList(control_mlp)

    def forward(self, x, *args):
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
            edge_types = self.batch_edge_types
        else:
            batch_size = x.shape[0]
            edge_index = self.get_batch_edge_index(batch_size)
            edge_types = self.get_batch_edge_types(batch_size)

        x = x.view(-1, self.node_features)
        # x = x.flatten(0, 1)

        if self.use_res:
            res0 = self.res_proj_0(x)
        x = F.dropout(F.relu(self.conv1(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)
        # x = self.norm1(x, batch, batch_size)

        if self.use_res:
            x = x + res0
            res1 = self.res_proj_1(x)
        x = F.dropout(F.relu(self.conv2(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)
        # x = self.norm2(x, batch, batch_size)

        if self.use_res:
            x = x + res1
            res2 = self.res_proj_2(x)
        x = F.dropout(F.relu(self.conv3(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)
        # x = self.norm3(x, batch, batch_size)

        if self.use_res:
            x = x + res2
            res3 = self.res_proj_3(x)
        x = F.dropout(F.relu(self.conv4(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)
        # x = self.norm4(x, batch, batch_size)

        if self.use_res:
            x = x + res3
            res4 = self.res_proj_4(x)
        x = F.dropout(F.relu(self.conv5(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)

        if self.use_res:
            x = x + res4
        #     res5 = self.res_proj_5(x)
        # x = F.dropout(F.relu(self.conv6(x, edge_index, edge_type=edge_types)), p=self.dropouts, training=self.training)
        #
        # if self.use_res:
        #     x = x + res5
        # unfold the batched graph
        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        for layer in self.gnn:
            th.nn.init.kaiming_uniform_(layer.lin.weight, mode="fan_in", nonlinearity="relu")
            if layer.lin.bias is not None:
                th.nn.init.zeros_(layer.lin.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp + self.control_mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
            th.nn.init.zeros_(layer.bias)

    def make_decision(self, x: th.Tensor):
        mlp_x = x
        control_x = x
        for layer in self.mlp[:-1]:
            mlp_x = F.leaky_relu(layer(mlp_x), negative_slope=0.05)
            # x = F.dropout(F.leaky_relu(layer(x)), p=0.5, training=self.training)

        if self.training:
            for layer in self.control_mlp[:-1]:
                control_x = F.leaky_relu(layer(control_x), negative_slope=0.05)

            return self.mlp[-1](mlp_x), self.control_mlp[-1](control_x)
        else:
            return self.mlp[-1](mlp_x)
