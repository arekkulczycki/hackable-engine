# -*- coding: utf-8 -*-

from numpy import float32

from typing import List

from hackable_engine.common.constants import INF
from hackable_engine.criteria.selection.base_selector import BaseSelector
from hackable_engine.game_tree.node import Node


class FastSelector(BaseSelector):
    """
    Selecting one best node fast.
    """

    def select(self, nodes: List[Node], color: bool) -> Node:
        best_node: Node
        best_score: float32 = -INF if color else INF

        for node in nodes:
            # node white-to-move then best child is one with the highest score
            score = node.score
            if color:
                if score >= best_score:
                    best_node = node
                    best_score = score
            else:
                if score <= best_score:
                    best_node = node
                    best_score = score

        return best_node
