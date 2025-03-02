# -*- coding: utf-8 -*-
import torch as th
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.sampler import BaseSampler, NodeSamplerInput, SamplerOutput, NeighborSampler
from torch_geometric.sampler.base import SubgraphType

from hackable_engine.training.device import Device


class HexGraphSampler(BaseSampler):

    def __init__(self, data, num_neighbors=10):
        """
        Custom neighbor sampler that selects neighbors based on feature similarity.

        Args:
            data (torch_geometric.data.Data): Input graph data.
            num_neighbors (int): Number of neighbors to sample per node.
        """
        super().__init__()
        self.data = data
        self.num_neighbors = num_neighbors

    def sample_from_nodes(self, index: NodeSamplerInput) -> SamplerOutput:
        """Proposition from ChatGPT"""
        edge_index = self.data.edge_index
        x = self.data.x  # Node features

        row, col = edge_index  # Extract source and target nodes
        sampled_edges = []
        # sampled_nodes = set(index.tolist())  # Keep track of nodes included in the subgraph
        sampled_nodes = index.node.tolist()

        for node in index.node:
            # Get neighbors of the current node
            neighbors = col[row == node]
            if len(neighbors) == 0:
                continue  # Skip if no neighbors

            # Compute feature similarity between node and its neighbors
            node_feat = x[node].unsqueeze(0)  # (1, F)
            neighbor_feats = x[neighbors]  # (num_neighbors, F)

            # Compute cosine similarity
            similarities = th.cosine_similarity(node_feat, neighbor_feats)

            # Convert similarities to sampling probabilities
            probs = th.softmax(similarities, dim=0)

            # Sample neighbors based on these probabilities
            num_sample = min(self.num_neighbors, len(neighbors))
            sampled_idx = th.multinomial(probs, num_sample, replacement=False)

            sampled_neighbors = neighbors[sampled_idx]
            sampled_edges.extend([(node, neighbor.item()) for neighbor in sampled_neighbors])
            sampled_nodes.update(sampled_neighbors.tolist())

        # Convert sampled edges to tensor
        if sampled_edges:
            sampled_edges = th.tensor(sampled_edges).T  # Shape (2, num_edges)
        else:
            sampled_edges = th.empty((2, 0), dtype=th.long)

        # Convert sampled nodes to tensor
        sampled_nodes = th.tensor(list(sampled_nodes), dtype=th.long)

        # Create a SamplerOutput object (ensuring `edge` is properly set)
        return SamplerOutput(
            node=sampled_nodes,  # Sampled node indices
            edge=sampled_edges,  # Sampled edges in (2, num_edges) format
            row=sampled_edges[0] if sampled_edges.numel() > 0 else th.empty(0, dtype=th.long),
            col=sampled_edges[1] if sampled_edges.numel() > 0 else th.empty(0, dtype=th.long),
            batch=th.arange(len(sampled_nodes))  # Dummy batch indices
        )

    def _sample(
        self,
        seed: th.Tensor,
        seed_time: th.Tensor | None = None,
        **kwargs,
    ) -> SamplerOutput:
        """Original sample function of NeighborSampler for `torch_geometric.typing.WITH_TORCH_SPARSE`."""
        out = th.ops.torch_sparse.neighbor_sample(
            self.colptr,
            self.row,
            seed,  # seed
            self.num_neighbors.get_mapped_values(),
            self.replace,
            self.subgraph_type != SubgraphType.induced,
        )
        node, row, col, edge, batch = out + (None,)
        num_sampled_nodes = num_sampled_edges = None
        return SamplerOutput(
            node=node,
            row=row,
            col=col,
            edge=edge,
            batch=batch,
            num_sampled_nodes=num_sampled_nodes,
            num_sampled_edges=num_sampled_edges,
        )


def get_batch(x, edge_index):
    # return edge_index
    # return th.cat(tuple(edge_index for _ in range(x.shape[0])), dim=1)
    b = next(
        iter(
            DataLoader(
                [Data(x=x_, edge_index=edge_index) for x_ in x],
                batch_size=x.shape[0],
            )
        )
    )
    b.batch = b.batch.to(Device.XPU)
    return b


def get_sample(x, edge_index, n_graph_layers):
    batch = get_batch(x, edge_index)
    # print("batch", batch)
    samples = (
        (s.x, s.edge_index)
        for s in iter(
            NeighborLoader(
                batch,
                batch_size=x.shape[0],
                num_neighbors=[-1 for _ in range(n_graph_layers)],
                shuffle=True,
            )
        )
    )
    sample_x, sample_edge_index = zip(*samples)
    return th.cat(sample_x), th.cat(sample_edge_index, dim=1)
