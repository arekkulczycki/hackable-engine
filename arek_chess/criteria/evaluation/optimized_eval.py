"""
Evaluation by all the attributes that can be obtained from board in an optimized way.

Desired:
[x] is_check
[x] material (with advanced pawn bonus)
[ ] color on which white pieces are
[ ] color on which black pieces are
[ ] color on which white pawns are
[ ] color on which black pawns are
[ ] mobility
[ ] threats (x ray)
[ ] king threats (x ray)
[ ] king proximity threats (direct)
[ ] king mobility (?)
[ ] protection
[ ] advancement
[ ] protection x advancement
[ ] pawn structure defined as a binary number
[ ]

Observation:
[ ] material on board
[ ] white king location
[ ] black king location
[ ] white forces location (density of attacks, 64 floats or simplified to just avg coordinates)
[ ] black forces location (density of attacks, 64 floats or simplified to just avg coordinates)
[ ] openness of the position (many pawns locked -> many pawns gone)
[ ] colors of remaining white bishops
[ ] colors of remaining black bishops
[ ]
"""

from typing import List

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.base_eval import BaseEval


class OptimizedEval(BaseEval):
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

        is_check = (
            -int(board.is_check()) if board.turn else int(board.is_check())
        )  # color is the one who gave the check
        material = board.get_material_simple(True) - board.get_material_simple(False)
        space = board.get_space(True) - board.get_space(False)
        params = [
            is_check,
            material,
            space,
        ]

        return board.calculate_score(action, params)
