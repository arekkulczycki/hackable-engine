# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM
from typing import Tuple, List

from pyinstrument import Profiler

from arek_chess.board.board import Board, Move
from arek_chess.utils.memory_manager import (
    MemoryManager,
    remove_shm_from_resource_tracker,
)
from arek_chess.utils.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker


class EvalWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.0005

    def __init__(self, eval_queue: Queue, selector_queue: Queue, action: List[float]):
        super().__init__()

        self.eval_queue = eval_queue
        self.selector_queue = selector_queue
        self.action = action

    def setup(self):
        """"""

        remove_shm_from_resource_tracker()

        # self.call_count = 0
        #
        # self.profile_code()

    def run(self):
        """

        :return:
        """

        self.setup()

        while True:
            eval_item = self.eval_queue.get()
            if eval_item:
                (
                    node_name,
                    size,
                    move,
                    turn_after,
                    captured_piece_type,
                ) = eval_item

                # self.call_count += 1

                # benchmark max perf with random generation
                # score = uniform(-10, 10)
                score = self.get_score(node_name, turn_after, move, captured_piece_type)

                self.selector_queue.put(
                    (
                        node_name,
                        size,
                        move,
                        turn_after,
                        captured_piece_type,
                        score,
                    )
                )
            else:
                time.sleep(self.SLEEP)

    def get_score(
        self,
        node_name: str,
        turn_after: bool,
        move_str: str,
        captured_piece_type: int,
    ) -> Tuple[float, str]:
        """

        :return:
        """

        board, move, moved_piece_type = self.get_move_data(
            move_str, node_name, not turn_after
        )

        params = MemoryManager.get_node_params(node_name)

        params[0] += board.get_material_delta(captured_piece_type)
        safety_w = board.get_safety_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        safety_b = board.get_safety_delta(
            False, move, moved_piece_type, captured_piece_type
        )
        under_w = board.get_under_attack_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        under_b = board.get_under_attack_delta(
            False, move, moved_piece_type, captured_piece_type
        )

        params[1] += safety_w - safety_b
        params[2] += under_w - under_b
        params[3] += board.get_mobility_delta(move, captured_piece_type)
        params[4] = board.len_empty_squares_around_king(
            True, move
        ) - board.len_empty_squares_around_king(False, move)
        params.append(board.get_king_threats(True) - board.get_king_threats(False))

        score = board.calculate_score(self.action, params, moved_piece_type)

        return score

    def get_move_data(
        self, move_str: str, node_name: str, turn_before: bool
    ) -> Tuple[Board, Move, int]:
        board = MemoryManager.get_node_board(node_name)
        board.turn = turn_before

        move = Move.from_uci(move_str)
        moving_piece_type = board.get_moving_piece_type(move)

        return board, move, moving_piece_type

    def profile_code(self):
        profiler = Profiler()
        profiler.start()

        def before_exit(*args):
            """"""
            print(f"call count: {self.call_count}")
            profiler.stop()
            profiler.print(show_all=True)
            # self.terminate()
            exit(0)

        signal(SIGTERM, before_exit)
