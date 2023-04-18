# -*- coding: utf-8 -*-
"""
Selecting randomly with higher probability the higher the score.
"""

from random import choices
from typing import List

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.game_tree.node import Node


class ExpProbabilitySelector(BaseSelector):
    """
    Selecting randomly with higher probability the higher the score.
    """

    @staticmethod
    def select(nodes: List[Node], color: bool) -> Node:
        best_node: Node
        normalized_weights: List[float] = ExpProbabilitySelector._normalized_weights(
            [float(node.score) for node in nodes], color
        )

        try:
            return choices(nodes, normalized_weights)[0]
        except ValueError:
            return nodes[0]

    @staticmethod
    def _normalized_weights(scores: List[float], color: bool) -> List[float]:
        """
        Normalize scores so that are all between 0 and 1.
        """

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return scores

        discount = max_score - min_score

        return (
            [((score - min_score) / discount) ** 3 for score in scores]
            if color
            else [(1 - ((score - min_score) / discount) ** 3) for score in scores]
        )
