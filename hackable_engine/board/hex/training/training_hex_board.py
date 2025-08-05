from __future__ import annotations

from typing import (
    Optional,
)

import numpy as np
import torch as th
from astar import find_path
from numpy import asarray, empty, int8, mean, zeros, ndarray
from torch_geometric.data import Data as GraphData, HeteroData

from hackable_engine.board import BitBoard
from hackable_engine.board.hex.bitboard_utils import (
    generate_masks,
)
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.board.hex.move import Move
from hackable_engine.board.hex.serializers import BoardShapeError
from hackable_engine.board.hex.types import EdgeType
from hackable_engine.common.constants import FLOAT_TYPE


class TrainingHexBoard(HexBoard):

    def __init__(self, *args, use_graph: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if use_graph and self.size > 1:
            # self.edge_index = th.tensor(list(self._get_all_graph_links()), dtype=th.long).t().contiguous()
            self.edge_index = self._get_all_graph_links_coo()
            self.edge_types = self._get_graph_link_types_one_hot()
            self.edge_types_rgcn = self._get_graph_link_types()
            self.coordinates = self._get_coordinates()
            pseudo_row, pseudo_col = self.edge_index
            self.pseudo_coordinates = self.coordinates[pseudo_col] - self.coordinates[pseudo_row]
        else:
            self.edge_index = th.tensor([])
            self.edge_types = th.tensor([])
            self.edge_types_rgcn = th.tensor([])
            self.coordinates = th.tensor([])
            self.pseudo_coordinates = th.tensor([])


    def _get_all_graph_links_coo(self) -> th.Tensor:
        """
        Get `edge_index` for graph data.
        """

        return (
            th.tensor(
                [
                    (link[0].bit_length() - 1, link[1].bit_length() - 1)
                    for link in self._get_all_graph_links()
                ],
                dtype=th.long,
            )
            .t()
            .contiguous()
        )

    def _get_all_graph_links(self) -> set[tuple[BitBoard, BitBoard]]:
        """
        Return all links between all board cells, considering the board to be a graph.
        """

        links: set[tuple[BitBoard, BitBoard]] = set()

        for mask in generate_masks(self.get_all_mask()):
            for neighbour_mask in self.generate_neighbours(mask):
                links.add((mask, neighbour_mask))
                # links.add((neighbour_mask, mask))  # this should always be added anyway within the outer loop

        return links

    def _get_graph_link_types(self) -> th.Tensor:
        """Returns edge types as tensor with shape (num_edges,)"""
        link_types: list[EdgeType] = []
        link: tuple[BitBoard, BitBoard]
        for from_, to_ in self.edge_index.t():
            from_x, from_y = Move.xy_from_mask(from_.item(), self.size)
            to_x, to_y = Move.xy_from_mask(to_.item(), self.size)
            if from_x == to_x:
                link_types.append(EdgeType.VERTICAL)
            elif from_y == to_y:
                link_types.append(EdgeType.HORIZONTAL)
            else:
                link_types.append(EdgeType.DIAGONAL)
        return th.tensor(link_types)

    def _get_graph_link_types_one_hot(self) -> th.Tensor:
        """Returns one-hot encoded edge types as tensor with shape (num_edges, 3)"""
        link_types = []
        link: tuple[BitBoard, BitBoard]
        for from_, to_ in self.edge_index.t():
            from_x, from_y = Move.xy_from_mask(from_.item(), self.size)
            to_x, to_y = Move.xy_from_mask(to_.item(), self.size)
            if from_x == to_x:
                link_types.append([1, 0, 0])  # EdgeType.VERTICAL)
            elif from_y == to_y:
                link_types.append([0, 1, 0])  # EdgeType.HORIZONTAL)
            else:
                link_types.append([0, 0, 1])  # EdgeType.DIAGONAL)
        return th.tensor(link_types).to(th.int)

    def _get_coordinates(self):
        return th.tensor([
            [row, col]
            for row in range(self.size)
            for col in range(self.size)
        ], dtype=th.float32)

    def to_homo_graph_data(self) -> GraphData:
        """"""

        return GraphData(
            x=th.from_numpy(self.get_homo_graph_node_features()),
            edge_index=self.edge_index,
        )

    def get_homo_graph_node_features(self) -> ndarray:
        """
        Get node features, where the only feature is stone color (or lack thereof).
        :return: tensor of shape (self.size_square, 1)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for _ in range(self.size_square):
            node_feature = 0
            if whites & 1:
                node_feature = 1
            elif blacks & 1:
                node_feature = -1

            node_features.append(node_feature)

            whites >>= 1
            blacks >>= 1

        return np.array([node_features], dtype=FLOAT_TYPE).transpose()

    def get_homo_graph_node_features_one_hot(self) -> ndarray:
        """
        Get node features, where the only feature is stone color (or lack thereof), but one-hot encoded.
        :return: tensor of shape (self.size_square, 3)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for _ in range(self.size_square):
            is_empty, is_white, is_black = 0, 0, 0
            if whites & 1:
                is_white = 1
            elif blacks & 1:
                is_black = 1
            else:
                is_empty = 1

            node_features.append((is_empty, is_white, is_black))

            whites >>= 1
            blacks >>= 1

        return np.array(node_features, dtype=FLOAT_TYPE)

    def to_hetero_graph_data(self) -> HeteroData:
        """"""

        return HeteroData(
            x=th.from_numpy(self.get_hetero_graph_node_features()),
            edge_index=self.edge_index,
        )

    def get_hetero_graph_node_embedding(self) -> ndarray:
        """
        Get node features, where the features are: stone color, edge of the board.
        :return: tensor of shape (self.size_square, 1)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for i in range(self.size_square):
            row = i // self.size
            col = i % self.size
            stone = 0 if blacks & 1 else 1 if whites & 1 else 2
            edge_white = 0 if col == 0 else 1 if col == (self.size - 1) else 2
            edge_black = 0 if row == 0 else 1 if row == (self.size - 1) else 2

            node_features.append((stone, edge_black, edge_white))

            whites >>= 1
            blacks >>= 1

        return np.array(node_features, dtype=FLOAT_TYPE).transpose()

    def _get_nodes_and_links(self) -> tuple[ndarray, ndarray]:
        """
        By convention the board graph has always all the nodes and link types, counting empty.

        Only the type of the node and link changes.
            - Available node types: empty, white, black.
            - Available link types: empty, white, black, mixed.

        To obtain a full graph a third value is required, which is constant per board - table of links.

        :returns: node types and link types
        """

        nodes = []
        links = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        last_color: int = 0
        for i in range(self.size_square):
            color: int = 0
            if whites & 1:
                color = 1
            elif blacks & 1:
                color = -1

            nodes.append(color)
            if i % self.size != 0:
                links.append(self._get_graph_link_type(last_color, color))

            last_color = color

            whites >>= 1
            blacks >>= 1

        self._append_graph_vertical_links_(nodes, links)
        self._append_graph_diagonal_links_(nodes, links)

        return asarray(nodes), asarray(links)

    def _append_graph_vertical_links_(self, nodes: list[int], links: list[int]):
        """
        Links along columns.

        TODO: think if better done same as `_append_graph_diagonal_links`
        """

        last_color: int = 0
        for k in range(self.size):
            for i in range(k, self.size_square + k, self.size):
                color = nodes[i]
                if i >= self.size:
                    # links along rows
                    links.append(self._get_graph_link_type(last_color, color))
                last_color = color

    def _append_graph_diagonal_links_(self, nodes: list[int], links: list[int]):
        """
        Links along the short diagonal.
        """

        for i in range(self.size_square):
            color = nodes[i]
            try:
                neighbour_mask = self.cell_downleft(1 << i)
                if not color or self.unoccupied & neighbour_mask:
                    links.append(0)

                elif color == 1 and self.occupied_co[True] & neighbour_mask:
                    links.append(1)
                elif color == 2 and self.occupied_co[False] & neighbour_mask:
                    links.append(2)
                else:
                    links.append(3)

            except BoardShapeError:
                continue

    @staticmethod
    def _get_graph_link_type(last_color: Optional[int], color: int) -> int:
        """"""

        if not last_color or not color:
            return 0

        elif last_color == 1 and color == 1:
            return 1

        elif last_color == 2 and color == 2:
            return 2

        return 3

