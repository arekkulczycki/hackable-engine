import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import LayerNorm

from hackable_engine.training.utils.device import Device
from hackable_engine.training.models import BaseModule


class GraphGAT(BaseModule):

    def __init__(
        self,
        node_count,
        node_features,
        output_size,
        batch_size,
        dropouts,
        num_envs,
        gnn_shape,
        gnn_heads,
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
        self.gnn_heads = (gnn_heads for _ in gnn_shape) if isinstance(gnn_heads, int) else gnn_heads
        self.mlp_shape = mlp_shape
        self.device = device

        self.edge_index = edge_index.to(device)
        self.batch_edge_index = self.get_batch_edge_index(batch_size)
        self.edge_types = edge_types.to(device).to(th.float32)
        self.batch_edge_types = self.get_batch_edge_types(batch_size)

        self.setup_graph_feature_extractor()
        self.setup_mlp(gnn_shape, mlp_shape, output_size)

        self.initialize_gnn_weights()
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
        return th.cat(batch_edge_types, dim=0).to(self.device)

    def setup_graph_feature_extractor(self):
        concats = [False for _ in self.gnn_shape]
        gnn = []
        residuals = []
        norms = []
        prev = self.node_features
        prev_heads = 1
        prev_concat = False
        for channels, heads, concat in zip(self.gnn_shape, self.gnn_heads, concats):
            if not prev_concat:
                prev_heads = 1

            gnn.append(GATv2Conv(prev * prev_heads, channels, heads=heads, concat=concat, edge_dim=3, add_self_loops=False))
            residuals.append(nn.Linear(prev * prev_heads, channels, device=self.device))
            norms.append(LayerNorm(channels))
            prev = channels
            prev_heads = heads
            prev_concat = concat

        self.gnn = nn.ModuleList(gnn)
        self.residuals = nn.ModuleList(residuals)
        self.norms = nn.ModuleList(norms)

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

        # for conv, residual in zip(self.gnn, self.residuals):
        #     h = conv(x, edge_index, edge_attr=edge_types)
        for conv, residual, norm in zip(self.gnn, self.residuals, self.norms):
            h = norm(conv(x, edge_index, edge_attr=edge_types))
            # x = F.dropout(F.relu(h + residual(x)), p=0.1, training=self.training)
            x = F.dropout(F.relu(h), p=self.dropouts, training=self.training) + residual(x)

        # unfold the batched graph
        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        for layer in self.gnn:
            th.nn.init.kaiming_uniform_(layer.lin_l.weight, mode="fan_in", nonlinearity="relu")
            th.nn.init.kaiming_uniform_(layer.lin_r.weight, mode="fan_in", nonlinearity="relu")

            if layer.lin_l.bias is not None:
                th.nn.init.zeros_(layer.lin_l.bias)

            if layer.lin_r.bias is not None:
                th.nn.init.zeros_(layer.lin_r.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp + self.control_mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
            th.nn.init.zeros_(layer.bias)

    def make_decision(self, x: th.Tensor):
        mlp_x = x
        control_x = x
        for layer in self.mlp[:-1]:
            mlp_x = F.leaky_relu(layer(mlp_x), negative_slope=0.05)
            # x = F.dropout(F.leaky_relu(layer(x)), p=self.dropouts, training=self.training)

        if self.training:
            for layer in self.control_mlp[:-1]:
                control_x = F.leaky_relu(layer(control_x), negative_slope=0.05)

            return self.mlp[-1](mlp_x), self.control_mlp[-1](control_x)
        else:
            return self.mlp[-1](mlp_x)
