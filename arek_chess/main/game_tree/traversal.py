# -*- coding: utf-8 -*-
"""
Tree traversal model.
"""

from typing import Optional, Dict

from arek_chess.main.game_tree.node.node import Node

from arek_chess.main.game_tree.node.create_node_mixin import CreateNodeMixin


class Traversal(CreateNodeMixin):
    """
    Tree traversal model.
    """

    root: Node
    tree: Dict[str, Node]

    def __init__(self, root: Node, tree: Dict[str, Node]):
        super().__init__()

        self.root = root
        self.tree = tree

        self.last_best_node = None

    def get_next_node_to_look_at(self) -> Optional[Node]:
        """"""

        best_node: Node = self.root

        k = 0
        while True:
            # if k == 1:
            #     if best_node != self.last_best_node:
            #         self.last_best_node = best_node
            #         print(f"top node: {best_node.name}, {best_node.score}")

            children = best_node.children
            if not children:
                # no children - TODO: handle if is checkmate
                best_node.being_processed = True
                return best_node

            children = [node for node in children if not node.being_processed]
            if children:
                best_node = self.select_promising_node(children, best_node.color)
            else:
                return None

            k += 1
