"""
Evaluation in the simplest manner, caring for just material and space in order
to balance risk/reward of pushing forward.
"""

from numpy import double

from arek_chess.board.legacy_board import Board
from arek_chess.criteria.evaluation.base_eval import BaseEval


class FastEval(BaseEval):
    """"""

    # material, space, is_check
    DEFAULT_ACTION: BaseEval.ActionType = (double(100.0), double(1.0), double(10.0))

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        action: BaseEval.ActionType = None
    ) -> double:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        material = board.get_material_simple(True) - board.get_material_simple(False)
        space = board.get_space(True) - board.get_space(False)
        is_check = (
            -int(board.is_check()) if board.turn else int(board.is_check())
        )  # color is the one who gave the check
        params = (
            double(material),
            double(space),
            double(is_check)
        )

        return board.calculate_score(action, params)
