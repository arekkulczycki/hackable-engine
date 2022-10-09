# -*- coding: utf-8 -*-
"""
Selecting one best node fast.
"""

from typing import List

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.main.game_tree.constants import INF
from arek_chess.main.game_tree.node.node import Node


class FastSelector(BaseSelector):
    """
    Selecting one best node fast.

    TODO: compile to C once the Node is reimplemented and this can work on both BaseNode and more complex child class
    """

    def select(self, nodes: List[Node], color: bool) -> Node:
        best_node: Node
        best_score: float = -INF if color else INF

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
