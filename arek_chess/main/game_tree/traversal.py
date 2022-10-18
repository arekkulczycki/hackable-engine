# -*- coding: utf-8 -*-
"""
Tree traversal model.
"""

from typing import Optional, List

from arek_chess.criteria.selection.base_selector import BaseSelector
from arek_chess.criteria.selection.fast_selector import FastSelector
from arek_chess.main.game_tree.node.node import Node


class Traversal:
    """
    Tree traversal model.
    """

    root: Node

    def __init__(self, root: Node):
        super().__init__()

        self.root = root

        self.selector: BaseSelector = FastSelector()

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
                # no children, because not yet received on the queue
                if best_node.being_processed:
                    return None

                # was looked at and no children - if the best bath leads to checkmate then don't select anything more
                if best_node.looked_at:
                    return None

                # is a leaf that hasn't yet been looked at
                best_node.looked_at = True
                best_node.being_processed = True
                return best_node

            children = [node for node in children if not node.being_processed]
            if children:
                best_node = self.select_promising_node(children, best_node.color)
            else:
                return None

            k += 1

    def select_promising_node(self, nodes: List[Node], color: bool) -> Node:
        """
        Get the child node, selected based on implemented criteria.
        """

        return self.selector.select(nodes, color)
