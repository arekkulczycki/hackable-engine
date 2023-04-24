# -*- coding: utf-8 -*-

from numpy import float32

from typing import List

from arek_chess.common.constants import INF
from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.game_tree.node import Node


class FastSelector(BaseSelector):
    """
    Selecting one best node fast.
    """

    @staticmethod
    def select(nodes: List[Node], color: bool) -> Node:
        node: Node
        score: float32
        best_node: Node
        best_score: float32 = -INF if color else INF

        for node in nodes:
            # node white-to-move then best child is one with the highest score
            score = node._score
            if color:
                if score >= best_score:
                    best_node = node
                    best_score = score
            else:
                if score <= best_score:
                    best_node = node
                    best_score = score

        return best_node
