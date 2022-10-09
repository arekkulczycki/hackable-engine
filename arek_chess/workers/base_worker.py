# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from multiprocessing import Process
from typing import Optional, Tuple

from arek_chess.board.board import Board
from arek_chess.board.board import Move
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.utils.memory_manager import MemoryManager


class BaseWorker(Process):
    """
    Base for the worker process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory_manager = MemoryManager()

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            exit(0)

    def _run(self):
        raise NotImplementedError

    def get_board_data(self, board: Optional[Board], node_name: str, move_str: str) -> Tuple[Board, int, int]:
        if not board:
            board = self.memory_manager.get_node_board(node_name)

        move = Move.from_uci(move_str)

        captured_piece_type = board.get_captured_piece_type(move)
        moved_piece_type = board.get_moving_piece_type(move)

        # board.light_push(move, state_required=True)
        board.push(move)

        return board, captured_piece_type, moved_piece_type

    def get_action(self, size: int) -> BaseEval.ActionType:
        """"""

        return self.memory_manager.get_action(size)
