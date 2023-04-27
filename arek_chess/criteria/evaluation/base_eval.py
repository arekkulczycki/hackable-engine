"""
Base class for node evaluation.
"""

from abc import ABC
from typing import Tuple, Callable, Optional

from numpy import float32

from arek_chess.board.board import Board
from numpy import dot

PENALIZER: float32 = float32(0.99)
REVERSE_PENALIZER: float32 = float32(1.01)

ActionType = Tuple[float32, ...]


class BaseEval(ABC):
    """
    Inherit from this class to implement your own evaluator.

    Provides calculation of the score in a given position on the board.
    In order to get information about the state of the board use the get_board_data method.
    Board class should provide all the board-specific or chess-specific logic about the current state of the game.

    Must implement just the get_score method.

    Any eval model should be designed for a specific training observation method.
    """

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        is_check: bool,
        action: Optional[ActionType] = None,
    ) -> float32:
        """
        :param board
        :param move_str:
        :param captured_piece_type: 0 - no capture, 1 - PAWN, 2 - KNIGHT, 3 - BISHOP, 4 - ROOK, 5 - QUEEN
        :param is_check:
        :param action:

        :return: score given to the candidate move in the current position
        """

        raise NotImplementedError

    @staticmethod
    def get_for_both_players(function: Callable[[bool], ActionType]) -> Tuple[float32, ...]:
        """"""

        return tuple(float32(a - b) for a, b in zip(function(True), function(False)))

    @staticmethod
    def calculate_score(
        action: ActionType, params: ActionType
    ) -> float32:
        score = dot(action, params)

        return score
