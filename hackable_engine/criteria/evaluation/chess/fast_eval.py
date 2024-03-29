"""
Evaluation in the simplest manner, caring for just material and space in order
to balance risk/reward of pushing forward.
"""

from numpy import float32

from hackable_engine.board.board import Board
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval


class FastEval(BaseEval):
    """"""

    # material, space, is_check
    DEFAULT_ACTION: WeightsType = (double(100.0), double(1.0), double(10.0))

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        action: WeightsType = None
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
