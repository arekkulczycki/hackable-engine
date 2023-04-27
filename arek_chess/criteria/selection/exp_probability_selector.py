# -*- coding: utf-8 -*-

from random import choices
from typing import List
from numpy import float32

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.game_tree.node import Node

ONE: float32 = float32(1)
EXPONENT: float32 = float32(3)


class ExpProbabilitySelector(BaseSelector):
    """
    Selecting randomly with higher probability the higher the score.
    """

    @staticmethod
    def select(nodes: List[Node], color: bool) -> Node:
        best_node: Node
        normalized_weights: List[float] = ExpProbabilitySelector._normalized_weights(
            [node.score for node in nodes], color
        )

        try:
            return choices(nodes, normalized_weights)[0]
        except ValueError:
            return nodes[0]

    @staticmethod
    def _normalized_weights(scores: List[float32], color: bool) -> List[float]:
        """
        Normalize scores so that are all between 0 and 1.
        """

        min_score = min(scores)
        max_score = max(scores)

        if min_score == max_score:
            return [1 for _ in scores]

        discount = max_score - min_score

        return (
            [float(((score - min_score) / discount) ** EXPONENT) for score in scores]
            if color
            else [float((ONE - ((score - min_score) / discount) ** EXPONENT)) for score in scores]
        )
