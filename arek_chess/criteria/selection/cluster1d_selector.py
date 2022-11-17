# -*- coding: utf-8 -*-
"""
Selecting one best node fast.
"""

from typing import List

import kmeans1d

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.criteria.selection.linear_probability_selector import LinearProbabilitySelector
from arek_chess.game_tree.node.node import Node

CLUSTER_2_3_THRESHOLD: int = 9
CLUSTER_3_4_THRESHOLD: int = 15
CLUSTER_4_5_THRESHOLD: int = 25


class Cluster1dSelector(BaseSelector):
    """
    Selecting randomly with higher probability the higher the score, but only from the top subset.
    """

    def select(self, nodes: List[Node], color: bool) -> Node:
        nnodes = len(nodes)
        k = (
            2
            if nnodes < CLUSTER_2_3_THRESHOLD
            else 3
            if nnodes < CLUSTER_3_4_THRESHOLD
            else 4
            if nnodes < CLUSTER_4_5_THRESHOLD
            else 5
        )  # number of clusters
        clusters, centroids = kmeans1d.cluster(
            [node.score for node in nodes], k
        )

        top_nodes = [node for node in nodes if node.score in clusters[0]]
        return LinearProbabilitySelector.select(top_nodes, color)
