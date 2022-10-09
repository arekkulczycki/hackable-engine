"""
Evaluation in the simplest manner, caring for just material and space in order
to balance risk/reward of pushing forward.
"""

from typing import List

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.base_eval import BaseEval


class FastEval(BaseEval):
    """"""

    # material, space, is_check
    DEFAULT_ACTION: List[float] = [100.0, 1.0, 10.0]

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
        is_check = (
            -int(board.is_check()) if board.turn else int(board.is_check())
        )  # color is the one who gave the check
        params = [
            material,
            space,
            is_check
        ]

        return board.calculate_score(action, params)
