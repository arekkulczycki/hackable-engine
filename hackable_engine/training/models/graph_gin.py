# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINConv

from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class GraphGIN(BaseModule):

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
        use_res = True,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.gnn_shape = gnn_shape
        self.mlp_shape = mlp_shape
        self.is_seq = is_seq
        self.use_res = use_res

        self.edge_index = edge_index.to(Device.XPU)
        self.batch_edge_index = self.get_batch_edge_index(node_count, batch_size)

        self.setup_graph_feature_extractor(gnn_shape)
        self.setup_mlp(gnn_shape, mlp_shape, output_size, is_seq)

        self.initialize_gnn_weights()
        self.initialize_mlp_weights()

    def get_batch_edge_index(self, node_count, batch_size):
        batch_edge_index = []
        for i in range(batch_size):
            batch_edge_index.append(self.edge_index + i * node_count)
        return th.cat(batch_edge_index, dim=1).to(Device.XPU)

    def setup_graph_feature_extractor(self, shape):
        if self.use_res:
            self.res_proj_0 = nn.Linear(self.node_features, shape[0], device=Device.XPU)
            # self.res_proj_01 = nn.Linear(self.node_features, shape[1], device=Device.XPU)
            # self.res_proj_02 = nn.Linear(self.node_features, shape[2], device=Device.XPU)
            self.res_proj_1 = nn.Linear(shape[0], shape[1], device=Device.XPU)
            self.res_proj_2 = nn.Linear(shape[1], shape[2], device=Device.XPU)
            self.res_proj_3 = nn.Linear(shape[2], shape[3], device=Device.XPU)
            self.res_proj_4 = nn.Linear(shape[3], shape[4], device=Device.XPU)
            self.res_proj_5 = nn.Linear(shape[4], shape[5], device=Device.XPU)

        # self.embedding = nn.Linear(self.node_features, shape[0])
        # nn.init.kaiming_uniform_(self.embedding.weight)
        # nn.init.zeros_(self.embedding.bias)

        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(self.node_features, shape[0]),
            nn.ReLU(),
            nn.Linear(shape[0], shape[0])
        ), train_eps=True)
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(shape[0], shape[1]),
            nn.ReLU(),
            nn.Linear(shape[1], shape[1])
        ), train_eps=True)
        self.conv3 = GINConv(nn.Sequential(
            nn.Linear(shape[1], shape[2]),
            nn.ReLU(),
            nn.Linear(shape[2], shape[2])
        ), train_eps=True)
        self.conv4 = GINConv(nn.Sequential(
            nn.Linear(shape[2], shape[3]),
            nn.ReLU(),
            nn.Linear(shape[3], shape[3])
        ), train_eps=True)
        self.conv5 = GINConv(nn.Sequential(
            nn.Linear(shape[3], shape[4]),
            nn.ReLU(),
            nn.Linear(shape[4], shape[4])
        ), train_eps=True)
        self.conv6 = GINConv(nn.Sequential(
            nn.Linear(shape[4], shape[5]),
            nn.ReLU(),
            nn.Linear(shape[5], shape[5])
        ), train_eps=True)
        self.gnn = (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6)


    def setup_mlp(self, gnn_shape, mlp_shape, output_size, is_seq = False):
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

        self.mlp = tuple(mlp)
        self.control_mlp = tuple(control_mlp)

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
        # x = self.embedding(x)

        if self.use_res:
            res0 = self.res_proj_0(x)
            # res01 = self.res_proj_01(x)
            # res02 = self.res_proj_02(x)
        x = F.dropout(F.relu(self.conv1(x, edge_index)), p=0.25, training=self.training)
        # x = self.norm1(x, batch, batch_size)

        if self.use_res:
            x = x + res0
            res1 = self.res_proj_1(x)
        x = F.dropout(F.relu(self.conv2(x, edge_index)), p=0.25, training=self.training)
        # x = self.norm2(x, batch, batch_size)

        if self.use_res:
            x = x + res1# + res01
            res2 = self.res_proj_2(x)
        x = F.dropout(F.relu(self.conv3(x, edge_index)), p=0.25, training=self.training)
        # x = self.norm3(x, batch, batch_size)

        if self.use_res:
            x = x + res2# + res02
            res3 = self.res_proj_3(x)
        x = F.dropout(F.relu(self.conv4(x, edge_index)), p=0.25, training=self.training)

        if self.use_res:
            x = x + res3# + res02
            res4 = self.res_proj_4(x)
        x = F.dropout(F.relu(self.conv5(x, edge_index)), p=0.25, training=self.training)

        if self.use_res:
            x = x + res4# + res02
            res5 = self.res_proj_5(x)
        x = F.dropout(F.relu(self.conv6(x, edge_index)), p=0.25, training=self.training)

        if self.use_res:
            x = x + res5# + res02
            # x = x + res4 + res02
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

    def initialize_mlp_weights(self):
        for layer in self.mlp + self.control_mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
            # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="tanh")
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
