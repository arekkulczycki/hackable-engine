# -*- coding: utf-8 -*-
from typing import Optional

from numpy import asarray, float32

from arek_chess.board.hex.hex_board import HexBoard
from arek_chess.criteria.evaluation.base_eval import ActionType, BaseEval, T

ONE: float32 = float32(1)


class SimpleEval(BaseEval[HexBoard]):
    """"""

    DEFAULT_ACTION: ActionType = (1.0, 1.0, 1.0, 1.0, 1.0)

    def get_score(
        self, board: T, is_check: bool, action: Optional[ActionType] = None
    ) -> float32:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        # sum connectedness measuring along two main directions
        connectedness, wingspan = board.get_connectedness_and_wingspan()

        # (black minus white) because negative of imbalance equals balance
        balance = board.get_imbalance(False) - board.get_imbalance(True)

        # local_pattern_black = board.get_local_pattern(False)
        # local_pattern_white = board.get_local_pattern(True)

        # TODO: predict function should get the patterns as input and produce eval and certainty as output
        # make certainty a tangens shape function
        local_pattern_eval = 0
        local_pattern_certainty = 1

        turn_bonus = ONE if board.turn else -ONE

        params = asarray((
            connectedness,
            balance,
            wingspan,
            local_pattern_eval * local_pattern_certainty,
            turn_bonus,
        ))

        return self.calculate_score(action, params)
