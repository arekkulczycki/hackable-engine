# -*- coding: utf-8 -*-
from typing import Optional

from numpy import asarray, float32, isclose

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval

ONE: float32 = float32(1)


class SimpleEval(BaseEval[HexBoard]):
    """"""

    PARAMS_NUMBER: int = 8
    DEFAULT_WEIGHTS: WeightsType = asarray(
        (
            float32(1.0),  # connectedness
            float32(1.0),  # spacing
            float32(5.0),  # balance
            float32(1.0),  # central_balance
            float32(15.0),  # missing distance - the value should roughly be equal to turn bonus
            # float32(5.0),  # two-dimensional entropy
            float32(15.0),  # turn bonus
            float32(0.0),  # local pattern eval
            float32(0.0),  # local pattern confidence
        ),
        dtype=float32,
    )

    def get_score(
        self, board: HexBoard, is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        """"""

        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        confidence: float32 = weights[-1]
        non_confidence: float32 = ONE - confidence

        turn_bonus = ONE if board.turn else -ONE

        if isclose(confidence, ONE):
            params = asarray((0, 0, 0, 0, 0, turn_bonus, ONE * 100), dtype=float32)

        else:
            # sum connectedness measuring along two main directions
            (
                connectedness_white,
                connectedness_black,
                spacing_white,
                spacing_black,
            ) = board.get_connectedness_and_spacing()

            bb, cb = board.get_imbalance(False)
            bw, cw = board.get_imbalance(True)

            # (black minus white) because missing distance the smaller the better
            if len(board.move_stack) > board.size_square / 4:
                missing_distance = board.get_shortest_missing_distance_perf(
                    False
                ) - board.get_shortest_missing_distance_perf(True)
            else:
                missing_distance = 0

            # w_entropy, b_entropy = 0, 0
            # if len(board.move_stack) > 10:
            #     w_entropy, _ = DispEn2D(board.color_matrix(True), 2)
            #     b_entropy, _ = DispEn2D(board.color_matrix(False), 2)

            # (black minus white) because negative of imbalance is balance
            balance: float32 = bb - bw
            central_balance: float32 = cb - cw

            params = asarray(
                (
                    (connectedness_white - connectedness_black) * non_confidence,
                    (spacing_white - spacing_black) * non_confidence,
                    balance * non_confidence,
                    central_balance * non_confidence,
                    missing_distance,  # float32((w_entropy - b_entropy) * 100),  # multiplying artificially to get a value > 1...
                    turn_bonus,
                    confidence * 100,  # multiply to rescale eval from 0-1 to 0-100
                ),
                dtype=float32,
            )

        return self.calculate_score(weights[:-1], params)
