# -*- coding: utf-8 -*-
from typing import Optional

from numpy import asarray, float32, isclose

from arek_chess.board.hex.hex_board import HexBoard
from arek_chess.criteria.evaluation.base_eval import ActionType, BaseEval

ONE: float32 = float32(1)


class SimpleEval(BaseEval[HexBoard]):
    """"""

    ACTION_SIZE: int = 8
    DEFAULT_ACTION: ActionType = asarray((
        float32(1.0),  # connectedness
        float32(1.0),  # wingspan
        float32(5.0),  # balance
        float32(1.0),  # central_balance
        float32(15.0),  # missing distance - the value should roughly be equal to turn bonus
        float32(15.0),  # turn bonus
        float32(0.0),  # local pattern eval
        float32(0.0),  # local pattern confidence
    ), dtype=float32)

    def get_score(
        self, board: HexBoard, is_check: bool, action: Optional[ActionType] = None
    ) -> float32:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        confidence: float32 = action[-1]
        non_confidence: float32 = ONE - confidence

        turn_bonus = ONE if board.turn else -ONE

        if isclose(confidence, ONE):
            params = asarray((0, 0, 0, 0, 0, turn_bonus, ONE * 100), dtype=float32)

        else:
            # sum connectedness measuring along two main directions
            (
                connectedness_white,
                connectedness_black,
                wingspan_white,
                wingspan_black,
            ) = board.get_connectedness_and_wingspan()

            bb, cb = board.get_imbalance(False)
            bw, cw = board.get_imbalance(True)

            # (black minus white) because missing distance the smaller the better
            missing_distance: int
            if len(board.move_stack) > board.size_square / 4:
                missing_distance = board.get_shortest_missing_distance(False) - board.get_shortest_missing_distance(
                    True)
            else:
                missing_distance = 0

            # (black minus white) because negative of imbalance equals balance
            balance: float32 = bb - bw
            central_balance: float32 = cb - cw

            params = asarray(
                (
                    (connectedness_white - connectedness_black) * non_confidence,
                    (wingspan_white - wingspan_black) * non_confidence,
                    balance * non_confidence,
                    central_balance * non_confidence,
                    missing_distance,
                    turn_bonus,
                    confidence * 100  # multiply to rescale eval from 0-1 to 0-100
                ),
                dtype=float32,
            )

        return self.calculate_score(action[:-1], params)
