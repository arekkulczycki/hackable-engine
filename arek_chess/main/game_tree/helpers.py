"""
Helper classes.
"""

from typing import Tuple, List

from anytree import Node

from arek_chess.main.game_tree.constants import Print


class GetBestMoveMixin:

    def _get_the_best_1st_level(self) -> Tuple[float, str]:
        sorted_children: List[Node] = sorted(
            self.root.children, key=lambda node: node.score, reverse=self.root.color
        )

        if self.PRINT == Print.CANDIDATES:
            for child in sorted_children:
                print(child.move, child.score)

        best = sorted_children[0]
        return best.score, best.move

    def get_best_move(self) -> Tuple[float, str]:
        """
        Get the first move with the highest score with respect to color.
        """

        return self._get_the_best_1st_level()
