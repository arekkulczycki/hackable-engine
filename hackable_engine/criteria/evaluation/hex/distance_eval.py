# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional

from numpy import asarray, float32

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval

ONE: float32 = float32(1)
BOARD_SIZE: int = 9
MAX_BALANCE: int = BOARD_SIZE * 3 // 4
MAX_DISTANCE_DIFF: int = 100


class DistanceEval(BaseEval[HexBoard]):
    """"""

    PARAMS_NUMBER: int = 3
    DEFAULT_WEIGHTS: WeightsType = asarray(
        (
            float32(1.0),  # balance
            float32(
                25.0
            ),  # missing distance - the value should roughly be equal to turn bonus
            float32(1.0),  # turn bonus
        ),
        dtype=float32,
    )

    def get_score(
        self, board: HexBoard, is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        """"""

        if weights is None:
            weights = self.DEFAULT_WEIGHTS

        turn_bonus = ONE if board.turn else -ONE

        bb, cb = board.get_imbalance(False)
        bw, cw = board.get_imbalance(True)

        # (black minus white) because negative of imbalance is balance
        balance: float32 = bb - bw
        central_balance: float32 = cb - cw
        relative_balance: float32 = (balance + central_balance) / MAX_BALANCE

        params = asarray(
            (
                relative_balance,
                self._get_whos_closer(board, len(board.move_stack)) / MAX_DISTANCE_DIFF,
                turn_bonus,
            ),
            dtype=float32,
        )

        return self.calculate_score(weights, params)

    @classmethod
    def _get_whos_closer(cls: DistanceEval, board: HexBoard, n_moves: int) -> float:
        """"""

        (
            black_missing,
            black_variants,
        ) = board.get_short_missing_distances(False)
        (
            white_missing,
            white_variants,
        ) = board.get_short_missing_distances(
            True, should_subtract=n_moves % 2 == 1
        )  # subtracts distance from white because (if) has 1 stone less on board

        black_score = sum(
            (cls._weight_distance(9 - k, n_moves) * v)
            for k, v in black_variants.items()
        )
        white_score = sum(
            (cls._weight_distance(9 - k, n_moves) * v)
            for k, v in white_variants.items()
        )

        return white_score - black_score

    @staticmethod
    def _weight_distance(distance, n_moves) -> float:
        """Calculate weighted value of distance. In the endgame close connections value more."""

        if n_moves > 30:
            return distance**3 / 81
        elif n_moves > 15:
            return distance**2 / 9
        else:
            return float(distance)
