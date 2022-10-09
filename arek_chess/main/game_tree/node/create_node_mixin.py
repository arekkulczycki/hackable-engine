# -*- coding: utf-8 -*-
"""
Provides a unified way of creating nodes.
"""

from typing import Optional, List

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.criteria.selection.fast_selector import FastSelector
from arek_chess.main.game_tree.node.node import Node


class CreateNodeMixin:
    """
    Provides a unified way of creating nodes.
    """

    def __init__(self, *args, **kwargs):
        self.selector: BaseSelector = FastSelector()

    def create_node(self, parent: Node, move, score, captured, level) -> Node:
        """"""

        parent_name = parent.name
        child_name = f"{parent_name}.{move}"
        color = level % 2 == 0 if self.root.color else level % 2 == 1

        node = Node(
            parent=parent,
            name=child_name,
            move=move,
            score=score,
            captured=captured,
            level=level,
            color=color,
        )

        self.tree[child_name] = node

        return node

    def get_best_node(self, nodes: List[Node], color: bool) -> Node:
        """"""

        best_node = nodes[0]
        best_score = best_node.score
        for node in nodes[1:]:
            child_score = node.score

            # node white-to-move then best child is one with the highest score
            if color:
                if child_score > best_score:
                    best_node = node
                    best_score = child_score
            else:
                if child_score < best_score:
                    best_node = node
                    best_score = child_score

        return best_node

    def select_promising_node(self, nodes: List[Node], color: bool) -> Node:
        """
        Get the child node, selected based on implemented criteria.
        """

        return self.selector.select(nodes, color)

    def get_best_child_node(self, node: Node) -> Optional[Node]:
        """
        Get the best child node.
        """

        children: List[Node] = node.children

        if not children:
            return None

        return self.get_best_node(children, node.color)
