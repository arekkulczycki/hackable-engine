from itertools import zip_longest
from typing import Optional, Tuple, Type

import gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GCNConv, ResGatedGraphConv, SAGEConv, FastRGCNConv, GATv2Conv, GCN2Conv, GraphConv, \
    GeneralConv

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.training.hyperparams import N_ENVS

GAT_HEADS = 6
CONV_GETTERS = {
    "simple": lambda in_f, out_f, *args: GraphConv(in_f, out_f, bias=False),
    "general": lambda in_f, out_f, *args: GeneralConv(in_f, out_f, bias=False),  # in-built residual approach
    "gcn": lambda in_f, out_f, *args: GCNConv(in_f, out_f, bias=False, normalize=False),
    "gcn2": lambda in_f, out_f, *args: GCN2Conv(in_f, out_f, bias=False, normalize=False),
    "res_gated": lambda in_f, out_f, *args: ResGatedGraphConv(in_f, out_f, bias=False),  # in-built residual approach
    "gat": lambda in_f, out_f, *args: GATv2Conv(in_f, out_f, heads=GAT_HEADS, bias=False),  # in-built residual approach
    "gat_1_head": lambda in_f, out_f, *args: GATv2Conv(in_f, out_f, bias=False),  # in-built residual approach
    "fast_rgcn": lambda in_f, out_f, num_relations: FastRGCNConv(in_f, out_f, num_relations=num_relations, aggr="sum", bias=False),
    "sage": lambda in_f, out_f, *args: SAGEConv(in_f, out_f, bias=False),
}


class HexGraphFeaturesExtractor(BaseFeaturesExtractor):
    """"""

    def __init__(
        self,
        observation_space: gym.Space,
        board_size: int,
        learning_rate: Optional[float] = None,
        output_filters: Tuple[int, ...] = (16,),
        conv_type: str = "gcn",
        activation_fn: Optional[Type[th.nn.Module]] = None,
        use_residuals: bool = False,
        should_initialize_weights: bool = False,
    ) -> None:
        device = th.device("xpu")
        node_features = board_size ** 2 if conv_type == "fast_rgcn" else 1
        """There is just one node feature which is the color of a stone: empty, white or black."""

        self.conv_type = conv_type
        self.activation_fns = [activation_fn() for _ in output_filters if activation_fn]
        self.use_residuals = use_residuals
        self.edge_index = HexBoard("", size=board_size, use_graph=True).edge_index.to(device=device)

        features_dim = board_size**2 * output_filters[-1]
        if self.conv_type == "gat":
            features_dim *= GAT_HEADS ** len(output_filters)
        if self.conv_type == "fast_rgcn":
            features_dim = output_filters[-1]

        super().__init__(observation_space, features_dim)  #, learning_rate)  # is this from my custom fork?

        conv_getter = CONV_GETTERS[self.conv_type]
        input_filters = [node_features, *output_filters[:-1]]
        self.convs = []

        gat_multiplier = 1
        for i, (in_f, out_f) in enumerate(zip(input_filters, output_filters)):
            # conv = GraphConvolution(in_f, out_f)
            conv = conv_getter(in_f * gat_multiplier, out_f * gat_multiplier, self.edge_index.size(1))
            setattr(self, f"conv_{i}", conv)  # has to be assigned to self to keep weights in GPU
            self.convs.append(conv)

            if self.conv_type == "gat":
                gat_multiplier *= GAT_HEADS

        if should_initialize_weights:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights for layers.
        """

        modules = self.convs
        n_modules = len(self.convs)

        for i in range(n_modules):
            module = modules[i]

            if isinstance(module, GCNConv):
                module = module.lin

            if (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.ReLU)) or (
                    n_modules > i + 1 and isinstance(modules[i + 1], th.nn.ReLU)
            ):
                th.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.LeakyReLU)) or (
                    n_modules > i + 1 and isinstance(modules[i + 1], th.nn.LeakyReLU)
            ):
                th.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.Tanhshrink)) or (
                    n_modules > i + 1 and isinstance(modules[i + 1], th.nn.Tanhshrink)
            ):
                th.nn.init.normal_(module.weight, mean=0, std=0.3)
            else:
                th.nn.init.xavier_normal_(module.weight)#, gain=0.5)

            if module.bias is not None:
                th.nn.init.zeros_(module.bias)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = observations.flatten().unsqueeze(1) if self.conv_type in ["gat", "gat_1_head"] else observations
        # x = observations.squeeze() if self.conv_type == "fast_rgcn" else observations.flatten().unsqueeze(1)
        # x = observations.flatten().unsqueeze(1) if self.conv_type in ["gat", "res_gated"] else observations.squeeze()

        conv: GraphConv
        for conv, af in zip_longest(self.convs, self.activation_fns):
            c = x
            x = conv(x, self.edge_index)

            if af:
                x = af(x)

            if self.use_residuals and conv.in_channels == conv.out_channels:
                x += c

        return th.reshape(x, (observations.size(0), self.features_dim))
