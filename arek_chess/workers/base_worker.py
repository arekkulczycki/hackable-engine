# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import sys
from multiprocessing import Process
from typing import Tuple, Dict, Optional

from chess import Move

from arek_chess.board.board import Board, SQUARE_NAMES, PIECE_SYMBOLS
from arek_chess.common.memory.shared_memory import remove_shm_from_resource_tracker
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.profiler_mixin import ProfilerMixin
from arek_chess.criteria.evaluation.base_eval import BaseEval


class BaseWorker(Process, ProfilerMixin):
    """
    Base for the worker process.
    """

    recycled_move: Move = Move.from_uci("a2a3")
    prev_board: Optional[Board]
    prev_state: Optional[Dict]

    def run(self) -> None:
        """"""

        try:
            self.memory_manager = MemoryManager()
            remove_shm_from_resource_tracker()

            self._run()
        except KeyboardInterrupt:
            sys.exit(0)

    def _run(self) -> None:
        """"""

        raise NotImplementedError

    def get_move(self, move_str: str) -> Move:
        """"""

        recycled_move = self.recycled_move

        recycled_move.from_square = SQUARE_NAMES.index(move_str[0:2])
        recycled_move.to_square = SQUARE_NAMES.index(move_str[2:4])
        recycled_move.promotion = PIECE_SYMBOLS.index(move_str[4]) if len(move_str) == 5 else None

        return recycled_move

    def get_board_data(self, board: Board, move_str: str) -> Tuple[Board, int, int]:
        """"""

        move = self.get_move(move_str)

        captured_piece_type = board.get_captured_piece_type(move)
        moved_piece_type = board.get_moving_piece_type(move)

        # board.light_push(move, state_required=True)
        self.prev_state = board.lighter_push(move)

        return board, captured_piece_type, moved_piece_type

    def get_action(self, size: int) -> BaseEval.ActionType:
        """"""

        return tuple(self.memory_manager.get_action(size))
