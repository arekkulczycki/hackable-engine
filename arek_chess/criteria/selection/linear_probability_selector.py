# -*- coding: utf-8 -*-
"""
Selecting randomly with higher probability the higher the score.
"""

from random import choices
from typing import List

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.main.game_tree.node.node import Node


class LinearProbabilitySelector(BaseSelector):
    """
    Selecting randomly with higher probability the higher the score.
    """

    def select(self, nodes: List[Node], color: bool) -> Node:
        best_node: Node
        scores: List[float] = [float(node.score) for node in nodes]
        edge_score: float = min(scores) if color else max(scores)
        weights: List[float] = (
            [score - edge_score for score in scores]
            if color
            else [edge_score - score for score in scores]
        )

        try:
            return choices(nodes, weights)[0]
        except ValueError:
            return nodes[0]
