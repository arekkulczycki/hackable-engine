# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from typing import Callable, Generic, Optional, Tuple, TypeVar

from nptyping import NDArray, Shape, Single
from numpy import dot, float32

from hackable_engine.board import GameBoardBase

PENALIZER: float32 = float32(0.99)
REVERSE_PENALIZER: float32 = float32(1.01)

WeightsType = NDArray[Shape, Single]
GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)


class BaseEval(ABC, Generic[GameBoardT]):
    """
    Inherit from this class to implement your own evaluator.

    Provides calculation of the score in a given position on the board.
    In order to get information about the state of the board use the get_board_data method.
    Board class should provide all the board-specific or chess-specific logic about the current state of the game.

    Must implement just the get_score method.

    Any eval model should be designed for a specific training observation method.
    """

    PARAMS_NUMBER: int
    
    @abstractmethod
    def get_score(
        self,
        board: GameBoardT,
        is_check: bool,
        weights: Optional[WeightsType] = None,
    ) -> float32:
        """
        :param board
        :param is_check:
        :param weights:

        :returns: score given to the candidate move in the current position
        """

        raise NotImplementedError

    @staticmethod
    def get_for_both_players(
        function: Callable[[bool], WeightsType]
    ) -> Tuple[float32, ...]:
        """"""

        return tuple(float32(a - b) for a, b in zip(function(True), function(False)))

    @staticmethod
    def calculate_score(weights: WeightsType, params: WeightsType) -> float32:
        score = dot(weights, params)

        return score
