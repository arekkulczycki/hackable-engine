import math
from typing import Optional

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval


class PathsEval(BaseEval[HexBoard]):
    """"""

    PARAMS_NUMBER: int = 0

    def __init__(self, size: int):
        """"""

        self.size = size
        self.size_square = size**2

    def get_score(self, board: HexBoard, is_check: bool, weights: Optional[WeightsType] = None) -> float:
        """"""

        return self._get_distance_score(board, self.size_square - board.unoccupied.bit_count())

    def _get_distance_score(self, board, n_moves: int) -> float:
        """
        Objective score, i.e. positive is white advantage, negative black advantage.
        Scaled to (-1, 1).

        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        function = (
            board.get_short_missing_distances_cached
            if n_moves >= 20 * self.size
            else board.get_short_missing_distances_perf_cached
        )
        white_missing, white_variants = function(
            True, should_subtract=n_moves % 2 == 1
        )  # subtracts distance from white because has 1 stone less on board, on odd moves
        black_missing, black_variants = function(False)

        white_score = sum((self._weight_distance(self.size - k, n_moves) * v) for k, v in white_variants.items())
        black_score = sum((self._weight_distance(self.size - k, n_moves) * v) for k, v in black_variants.items())

        if not white_score and not black_score:
            return 0.0

        return math.tanh((white_score - black_score) / (white_score + black_score))

    def _weight_distance(self, distance, n_moves) -> int:
        """Calculate weighted value of distance. In the endgame close connections value more."""

        if n_moves > self.size_square / 2:
            return distance**3 / self.size_square
        elif n_moves > self.size_square / 4:
            return distance**2 / self.size
        else:
            return distance
