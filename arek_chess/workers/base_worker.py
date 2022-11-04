# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import sys
import tracemalloc
from pyinstrument import Profiler
from multiprocessing import Process
from signal import signal, SIGTERM
from typing import Tuple

from chess import Move

from arek_chess.board.board import Board
from arek_chess.common.memory.shared_memory import remove_shm_from_resource_tracker
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.criteria.evaluation.base_eval import BaseEval


class BaseWorker(Process):
    """
    Base for the worker process.
    """

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

    @staticmethod
    def get_board_data(board: Board, move_str: str) -> Tuple[Board, int, int]:
        """"""

        move = Move.from_uci(move_str)

        captured_piece_type = board.get_captured_piece_type(move)
        moved_piece_type = board.get_moving_piece_type(move)

        # board.light_push(move, state_required=True)
        board.push(move)

        return board, captured_piece_type, moved_piece_type

    def get_action(self, size: int) -> BaseEval.ActionType:
        """"""

        return tuple(self.memory_manager.get_action(size))

    def profile_code(self) -> None:
        """"""

        profiler = Profiler()
        profiler.start()

        # tracemalloc.start()

        def before_exit(*_) -> None:
            """"""

            # print(f"call count: {self.call_count}")
            profiler.stop()
            profiler.print(show_all=True)

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            # tracemalloc.stop()

            sys.exit(0)

        signal(SIGTERM, before_exit)
