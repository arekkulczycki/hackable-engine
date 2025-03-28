# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.nn.norm import GraphNorm

from hackable_engine.common.constants import TH_FLOAT_TYPE
from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class GraphGINE(BaseModule):

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
        is_seq=False,
        use_res=True,
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
        self.edge_types = (
            nn.functional.one_hot(edge_types, num_classes=3)
            .to(Device.XPU)
            .to(TH_FLOAT_TYPE)
        )

        self.setup_graph_feature_extractor(gnn_shape)
        self.setup_mlp(gnn_shape, mlp_shape, output_size, is_seq)

        self.initialize_gnn_weights()
        self.initialize_mlp_weights()

    def setup_graph_feature_extractor(self, shape):
        if self.use_res:
            self.res_proj_0 = (
                lambda x: x
            )  # nn.Linear(self.node_features, shape[0], device=Device.XPU)
            self.res_proj_1 = (
                lambda x: x
            )  # nn.Linear(shape[0], shape[1], device=Device.XPU)
            self.res_proj_2 = (
                lambda x: x
            )  # nn.Linear(shape[1], shape[2], device=Device.XPU)
            self.res_proj_3 = (
                lambda x: x
            )  # nn.Linear(shape[2], shape[3], device=Device.XPU)
            self.res_proj_4 = (
                lambda x: x
            )  # nn.Linear(shape[2], shape[3], device=Device.XPU)
        # self.embedding = nn.Embedding(self.node_features, shape[0])
        self.embedding = nn.Linear(self.node_features, shape[0])
        nn.init.kaiming_uniform_(self.embedding.weight)
        nn.init.zeros_(self.embedding.bias)
        self.conv1 = GINEConv(
            nn.Sequential(
                nn.Linear(shape[0], shape[0]), nn.ReLU(), nn.Linear(shape[0], shape[0])
            ),
            edge_dim=3,
        )
        self.conv2 = GINEConv(
            nn.Sequential(
                nn.Linear(shape[0], shape[1]), nn.ReLU(), nn.Linear(shape[1], shape[1])
            ),
            edge_dim=3,
        )
        self.conv3 = GINEConv(
            nn.Sequential(
                nn.Linear(shape[1], shape[2]), nn.ReLU(), nn.Linear(shape[2], shape[2])
            ),
            edge_dim=3,
        )
        self.conv4 = GINEConv(
            nn.Sequential(
                nn.Linear(shape[2], shape[3]), nn.ReLU(), nn.Linear(shape[3], shape[3])
            ),
            edge_dim=3,
        )
        self.conv5 = GINEConv(
            nn.Sequential(
                nn.Linear(shape[3], shape[4]), nn.ReLU(), nn.Linear(shape[4], shape[4])
            ),
            edge_dim=3,
        )
        self.gnn = (self.conv1, self.conv2, self.conv3, self.conv4, self.conv5)
        self.norm1 = GraphNorm(shape[0])
        self.norm2 = GraphNorm(shape[1])
        self.norm3 = GraphNorm(shape[2])
        self.norm4 = GraphNorm(shape[3])

        self.batch_envs = th.repeat_interleave(
            th.arange(self.num_envs), self.node_count
        ).to(Device.XPU)
        # self.batch_envs_expanded = batch_envs.unsqueeze(-1).expand(-1, shape[-1])
        self.batch = th.repeat_interleave(
            th.arange(self.batch_size), self.node_count
        ).to(Device.XPU)
        # self.batch_expanded = batch.unsqueeze(-1).expand(-1, shape[-1])

    def setup_mlp(self, gnn_shape, mlp_shape, output_size, is_seq=False):
        mlp = []
        prev_size = gnn_shape[-1] * self.node_count if is_seq else gnn_shape[-1]
        for size in [*mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=Device.XPU))
            prev_size = size

        self.mlp = tuple(mlp)

    def forward(self, x, *args):
        x = x.flatten(0, 1)
        x = self.extract_features(x)
        if self.is_seq:
            x = x.flatten(-2, -1)
        x = self.make_decision(x)
        return x.flatten(1, -1)

    def extract_features(self, x):
        batch_size = x.shape[0]
        batch = self.batch if batch_size == self.batch_size else self.batch_envs

        x = x.view(-1, self.node_features)
        x = self.embedding(x)

        if self.use_res:
            res0 = self.res_proj_0(x)
        x = F.dropout(F.relu(self.conv1(x, self.edge_index, edge_attr=self.edge_types)), p=0.2)
        x = self.norm1(x, batch, batch_size)

        if self.use_res:
            x = x + res0
            res1 = self.res_proj_1(x)
        x = F.dropout(F.relu(self.conv2(x, self.edge_index, edge_attr=self.edge_types)), p=0.2)
        x = self.norm2(x, batch, batch_size)

        if self.use_res:
            x = x + res1
            res2 = self.res_proj_2(x)
        x = F.dropout(F.relu(self.conv3(x, self.edge_index, edge_attr=self.edge_types)), p=0.2)
        x = self.norm3(x, batch, batch_size)

        if self.use_res:
            x = x + res2
            res3 = self.res_proj_3(x)
        x = F.dropout(F.relu(self.conv4(x, self.edge_index, edge_attr=self.edge_types)), p=0.2)
        x = self.norm4(x, batch, batch_size)

        if self.use_res:
            x = x + res3
            res4 = self.res_proj_4(x)
        x = F.dropout(F.relu(self.conv5(x, self.edge_index, edge_attr=self.edge_types)), p = 0.2)

        if self.use_res:
            x = x + res4
        # unfold the batched graph
        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        gin: GINEConv
        for gin in self.gnn:
            for layer in gin.nn:
                if isinstance(layer, nn.Linear):
                    th.nn.init.kaiming_normal_(
                        layer.weight, mode="fan_in", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        th.nn.init.zeros_(layer.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
            # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="tanh")
            th.nn.init.zeros_(layer.bias)

    def make_decision(self, x: th.Tensor):
        for layer in self.mlp[:-1]:
            x = F.leaky_relu(layer(x))
        return self.mlp[-1](x)
