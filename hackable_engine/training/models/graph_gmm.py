import torch as th
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GMMConv, AttentionalAggregation
from torch_geometric.nn.norm import LayerNorm, GraphNorm

from hackable_engine.training.utils.device import Device
from hackable_engine.training.models import BaseModule


class GraphGMM(BaseModule):

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
        pseudo_coordinates,
        kernel_size=6,
        device=Device.XPU,
        is_actor_critic=False,
        should_initialize_weights=True,
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
        self.kernel_size = kernel_size
        self.is_actor_critic = is_actor_critic

        self.edge_index = edge_index.to(device)
        self.batch_edge_index = self.get_batch_edge_index(batch_size)
        self.pseudo_coordinates = pseudo_coordinates.to(device)
        self.batch_pseudo_coordinates = self.get_batch_pseudo_coord(batch_size)

        self.setup_graph_feature_extractor()
        self.setup_mlp(gnn_shape, mlp_shape, output_size)

        if should_initialize_weights:
            self.initialize_gnn_weights()
            self.initialize_mlp_weights()

    def get_batch_edge_index(self, batch_size):
        batch_edge_index = []
        for i in range(batch_size):
            batch_edge_index.append(self.edge_index + i * self.node_count)
        return th.cat(batch_edge_index, dim=1).to(self.device)

    def get_batch_pseudo_coord(self, batch_size):
        batch_pseudo_coordinates = []
        for i in range(batch_size):
            batch_pseudo_coordinates.append(self.pseudo_coordinates)
        return th.cat(batch_pseudo_coordinates, dim=0).to(self.device)

    def setup_graph_feature_extractor(self):
        gnn = []
        residuals = []
        # norms = []
        prev = self.node_features
        for channels in self.gnn_shape:
            gnn.append(GMMConv(prev, channels, dim=2, kernel_size=self.kernel_size))
            residuals.append(nn.Linear(prev, channels, device=self.device))
            # norms.append(LayerNorm(channels, device=th.device(self.device)))
            # norms.append(GraphNorm(channels, device=th.device(self.device)))
            prev = channels

        self.gnn = nn.ModuleList(gnn)
        self.residuals = nn.ModuleList(residuals)
        # self.norms = nn.ModuleList(norms)

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

        if self.is_actor_critic:
            pooling = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.gnn_shape[-1], self.gnn_shape[-1]), nn.ReLU(), nn.Linear(self.gnn_shape[-1], 1)
                )
            ).to(self.device)
            self.value_head = nn.Sequential(
                pooling,
                nn.LayerNorm(gnn_shape[-1]),
                nn.Linear(gnn_shape[-1], mlp_shape[-1]),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(mlp_shape[-1], 1),
            )
            self.policy_head = nn.Sequential(
                # nn.Linear(gnn_shape[-1], 1),
                nn.Linear(gnn_shape[-1], mlp_shape[0]),
                nn.LeakyReLU(negative_slope=0.05),
                nn.Linear(mlp_shape[0], 1),
            )
        else:
            self.mlp = nn.ModuleList(mlp)
            self.control_mlp = nn.ModuleList(control_mlp)

    def forward(self, x, *args):
        # with th.amp.autocast(Device.XPU, dtype=th.bfloat16, enabled=True):
        x = self.extract_features(x)
        # x = self.pooling(x, None)
        # x = self.pooling(x, self.get_batch_ids(self.node_count, x.size(0), self.device))

        if self.is_actor_critic:
            return self.get_policy_and_value(x)
        else:
            if self.training:
                x, control = self.make_decision(x)
                return x.flatten(1, -1), control
            else:
                return self.make_decision(x).flatten(1, -1)

    def extract_features(self, x):
        if x.size(0) == self.batch_size:
            batch_size = self.batch_size
            edge_index = self.batch_edge_index
            pseudo_coordinates = self.batch_pseudo_coordinates
        else:
            batch_size = x.size(0)
            edge_index = self.get_batch_edge_index(batch_size)
            pseudo_coordinates = self.get_batch_pseudo_coord(batch_size)

        x = x.view(batch_size * self.node_count, self.node_features)

        # for conv, residual, norm in zip(self.gnn, self.residuals, self.norms):
        for conv, residual in zip(self.gnn, self.residuals):
            # for conv, residual in zip(self.gnn, self.residuals):
            h = conv(x, edge_index, pseudo_coordinates)
            # h = norm(h)
            # x = F.dropout(F.relu(h + residual(x)), p=0.1, training=self.training)
            x = F.dropout(F.relu(h), p=self.dropouts, training=self.training) + residual(x)

        return x.view(batch_size, self.node_count, self.gnn_shape[-1])

    def initialize_gnn_weights(self):
        # the PyG stuff is initialized correctly internally
        # for layer in self.gnn:
        #     th.nn.init.kaiming_uniform_(
        #         layer.lin.weight, mode="fan_in", nonlinearity="relu"
        #     )
        #     if layer.lin.bias is not None:
        #         th.nn.init.zeros_(layer.lin.bias)

        for layer in self.residuals:
            if isinstance(layer, nn.Linear):
                th.nn.init.zeros_(layer.weight)
                th.nn.init.zeros_(layer.bias)
                # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")

    def initialize_mlp_weights(self):
        if self.is_actor_critic:
            for head in [self.value_head, self.policy_head]:
                for layer in head:
                    if isinstance(layer, nn.Linear):
                        th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="leaky_relu")
                        if layer.bias is not None:
                            th.nn.init.zeros_(layer.bias)  # Optionally initialize biases to 0
        else:
            for layer in self.mlp + self.control_mlp:
                th.nn.init.kaiming_normal_(
                    layer.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
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

    def get_policy_and_value(self, x: th.Tensor):
        logits = self.policy_head(x).flatten(1, -1)
        value = self.value_head(x).flatten(1, -1)

        return logits, value

    def forward_value(self, x):
        x = self.extract_features(x)
        return self.value_head(x).flatten(1, -1)

    @staticmethod
    def get_batch_ids(node_count, batch_size, device):
        return th.repeat_interleave(
            th.arange(batch_size), node_count
        ).to(device)
