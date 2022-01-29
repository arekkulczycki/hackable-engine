# -*- coding: utf-8 -*-
"""
Evaluation in the simplest manner, caring for just material and space in order
to balance risk/reward of pushing forward.
"""

from typing import List

from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.utils.memory_manager import MemoryManager


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
        action: List[float] = None,
    ) -> float:
        """

        :return:
        """

        if action is None:
            action = self.DEFAULT_ACTION

        board, move = self.get_board_data(move_str, node_name, color)

        # params = MemoryManager.get_node_params(node_name)

        material = board.get_material_simple(True) - board.get_material_simple(False)
        space = board.get_space(True) - board.get_space(False)
        params = [
            material,
            space,
            material * space,
        ]

        return board.calculate_score(action, params)
