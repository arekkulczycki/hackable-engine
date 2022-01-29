# -*- coding: utf-8 -*-
"""
Helper classes.
"""

from typing import Optional, Tuple, List

from anytree import Node, LevelOrderIter

from arek_chess.main.game_tree.constants import Print
from arek_chess.utils.memory_manager import MemoryManager


class GetBestMoveMixin:
    @staticmethod
    def _remove_stored_node_memory(node: Node):
        if node.is_leaf:
            try:
                MemoryManager.remove_node_board_memory(node.name)
            except FileNotFoundError:
                pass
        else:
            try:
                MemoryManager.remove_node_memory(node.name)
            except FileNotFoundError:
                pass

    @staticmethod
    def _skip_unused(node: Node, depth: int) -> bool:
        if node.level >= depth and getattr(node, "deep", False):
            return True
        else:
            node.parent.deep = True
            return False

    @staticmethod
    def _assign_score_to_parent(
        node: Node, current_parent: Optional[Node], starting_color: bool
    ) -> None:
        parent = node.parent

        # we're iteration in order so the first time is else parent we assign it
        if parent != current_parent:
            current_parent = parent
            current_parent.score = node.score

        else:
            # color==True then the highest score is to be chosen
            color = node.level % 2 == 1 if starting_color else node.level % 2 == 0

            if (color and node.score > current_parent.score) or (
                not color and node.score < current_parent.score
            ):
                current_parent.score = node.score

    def _get_the_best_1st_level(self, root: Node, starting_color: bool) -> Tuple[float, str]:
        sorted_children: List[Node] = sorted(
            root.children, key=lambda node: node.score, reverse=starting_color
        )

        if self.PRINT == Print.CANDIDATES:
            print(sorted_children)

        best = sorted_children[0]
        return best.score, best.move

    def get_best_move(
        self, root: Node, depth: int, starting_color: bool
    ) -> Tuple[float, str]:
        """
        walk over all nodes reversed and
          - remove all created memory
          - discard branches that should not be included
          - backprop score again from only non-pruned branches
        """

        current_parent = None
        node: Node

        ordered_nodes = [node for node in LevelOrderIter(root)]
        for node in reversed(ordered_nodes):
            self._remove_stored_node_memory(node)

            node_level = node.level
            if node_level < 1:
                break

            if self._skip_unused(node, depth):
                continue

            self._assign_score_to_parent(node, current_parent, starting_color)

        return self._get_the_best_1st_level(root, starting_color)
