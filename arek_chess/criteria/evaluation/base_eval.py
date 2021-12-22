# -*- coding: utf-8 -*-

from abc import ABC
from typing import List, Tuple

from arek_chess.board.board import Board, Move
from arek_chess.utils.memory_manager import MemoryManager


class BaseEval(ABC):
    """
    Inherit from this class to implement your own evaluator.

    Provides calculation of the score in a given position on the board.
    In order to get information about the state of the board use the get_board_data method.
    Board class should provide all the board-specific or chess-specific logic about the current state of the game.

    Must implement just the get_score method.
    """

    def get_score(
        self,
        action: List[float],
        node_name: str,
        color: bool,
        move_str: str,
        captured_piece_type: int,
    ) -> float:
        """

        :param node_name:
        :param color:
        :param move_str:
        :param captured_piece_type: 0 - no capture, 1 - PAWN, 2 - KNIGHT, 3 - BISHOP, 4 - ROOK, 5 - QUEEN

        :return: score given to the candidate move in the current position
        """

        raise NotImplementedError

    def get_board_data(
        self, move_str: str, node_name: str, turn_before: bool
    ) -> Tuple[Board, Move]:
        board = MemoryManager.get_node_board(node_name)
        board.turn = turn_before

        move = Move.from_uci(move_str)

        return board, move
