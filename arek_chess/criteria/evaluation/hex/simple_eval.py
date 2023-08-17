# -*- coding: utf-8 -*-
from typing import Optional

from numpy import float32, subtract

from arek_chess.board.hex.hex_board import HexBoard
from arek_chess.criteria.evaluation.base_eval import ActionType, BaseEval, T


class SimpleEval(BaseEval[HexBoard]):
    """"""

    DEFAULT_ACTION: ActionType = (1.0, 1.0, 1.0)

    def get_score(
        self, board: T, is_check: bool, action: Optional[ActionType] = None
    ) -> float32:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        # sum connectedness measuring along two main directions
        connectedness, wingspan = subtract(
            board.get_connectedness_and_wingspan(False),
            board.get_connectedness_and_wingspan(True),
        )
        balance = board.get_balance(False) - board.get_balance(True)

        # local_pattern_black = board.get_local_pattern(False)
        # local_pattern_white = board.get_local_pattern(True)

        # TODO: predict function should get the patterns as input and produce eval and certainty as output
        # make certainty a tangens shape function
        local_pattern_eval = 0
        local_pattern_certainty = 1

        params = (
            connectedness,
            balance,
            wingspan,
            local_pattern_eval * local_pattern_certainty,
        )

        return self.calculate_score(action, params)
