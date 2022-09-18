"""
Evaluation in the simplest manner, caring for just material and space in order
to balance risk/reward of pushing forward.
"""

from typing import List, Optional

from arek_chess.board.board import Board
from arek_chess.board.mypy_chess import Move
from arek_chess.criteria.evaluation.base_eval import BaseEval


class FastEval(BaseEval):
    """"""

    # material, space
    DEFAULT_ACTION: List[float] = [100.0, 1.0, 1.0]

    def get_score(
        self,
        node_name: str,
        color: bool,
        move_str: str,
        captured_piece_type: int,
        board: Optional[Board],
        action: List[float] = None
    ) -> float:
        """

        :return:
        """

        if action is None:
            action = self.DEFAULT_ACTION

        if not board:
            board, move = self.get_board_data(move_str, node_name, color)
        # else:
        #     move = Move.from_uci(move_str)

        # params = MemoryManager.get_node_params(node_name)

        material = board.get_material_simple(True) - board.get_material_simple(False)
        space = board.get_space(True) - board.get_space(False)
        params = [
            material,
            space,
            material * space,
        ]

        return board.calculate_score(action, params)
