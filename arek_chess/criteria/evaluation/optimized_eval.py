"""
Evaluation by all the attributes that are obtained from board in an optimized way.

Desired:
[x] is_check
[x] material
[ ] mobility
[ ] king mobility
[ ] threats (x ray included)
[ ] king threats
[ ] protection
[ ] advancement
[ ] protection x advancement
[ ] pawn structure defined as a binary number
[ ]
"""

from typing import List

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.base_eval import BaseEval


class FastEval(BaseEval):
    """"""

    # is_check, material
    DEFAULT_ACTION: List[float] = [10.0, 100.0, 1.0]

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        action: List[float] = None
    ) -> float:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        material = board.get_material_simple(True) - board.get_material_simple(False)
        space = board.get_space(True) - board.get_space(False)
        params = [
            material,
            space,
        ]

        return board.calculate_score(action, params)
