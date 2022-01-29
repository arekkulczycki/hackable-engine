# -*- coding: utf-8 -*-

from typing import List

from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.utils.memory_manager import MemoryManager


class ArekEval(BaseEval):
    """"""

    # material, safety, under_attack, mobility, king_mobility, king_threats
    DEFAULT_ACTION: List[float] = [100.0, 1.0, -1.0, 1.0, -1.0, 2.0]

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
        moved_piece_type = board.get_moving_piece_type(move)

        params = MemoryManager.get_node_params(node_name)

        params[0] += board.get_material_delta(captured_piece_type)
        safety_w = board.get_safety_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        safety_b = board.get_safety_delta(
            False, move, moved_piece_type, captured_piece_type
        )
        under_w = board.get_under_attack_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        under_b = board.get_under_attack_delta(
            False, move, moved_piece_type, captured_piece_type
        )

        params[1] += safety_w - safety_b
        params[2] += under_w - under_b
        params[3] += board.get_mobility_delta(move, captured_piece_type)
        try:
            params.append(board.get_king_threats(True) - board.get_king_threats(False))
        except:
            print(board.fen(), move_str, node_name)
            raise ValueError(f"incorrect pos: {board.fen()}, {move_str}, {node_name}")
        # params[4] += board.get_empty_squares_around_king_delta(move, captured_piece_type != 0)
        params[4] = board.get_king_mobility(True, move) - board.get_king_mobility(
            False, move
        )

        score = board.calculate_score(action, params, moved_piece_type)

        return score
