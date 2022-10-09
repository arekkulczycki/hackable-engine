# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM

from faster_fifo import Full
from pyinstrument import Profiler

from arek_chess.criteria.evaluation.legacy_eval import LegacyEval
from arek_chess.criteria.evaluation.fast_eval import FastEval
from arek_chess.main.game_tree.constants import INF
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager
from arek_chess.workers.base_worker import BaseWorker


class EvalWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.0001

    def __init__(self, input_queue: QueueManager, output_queue: QueueManager, constant_action: bool = False):
        super().__init__()

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.constant_action = constant_action

    def setup(self):
        """"""

        # remove_shm_from_resource_tracker()

        # self.call_count = 0

        # self.profile_code()

        self.evaluator = LegacyEval()
        # self.evaluator = FastEval()
        self.memory_manager = MemoryManager()

        self.max_items_at_once = 16

    def _run(self):
        """

        :return:
        """

        self.setup()

        while True:
            eval_items = self.input_queue.get_many(self.max_items_at_once)
            if eval_items:
                self.eval_items(eval_items)
            else:
                time.sleep(self.SLEEP)

    def eval_items(self, eval_items):
        """"""

        put_items = []
        to_set = {}

        names = [item[0] for item in eval_items]
        boards = self.memory_manager.get_many_boards(names)
        for eval_item, board in zip(eval_items, boards):
            (
                node_name,
                size,
                move_str,
            ) = eval_item

            # self.call_count += 1

            board, captured_piece_type, moved_piece_type = self.get_board_data(board, node_name, move_str)  # board after the move
            to_set[f"{node_name}.{move_str}"] = board

            if board.is_game_over(claim_draw=True):
                winner = board.outcome(claim_draw=True).winner
                score = 0 if winner is None else INF if winner else -INF

            else:
                action = None if self.constant_action else self.get_action(self.evaluator.ACTION_SIZE)
                score = self.evaluator.get_score(board, move_str, captured_piece_type, action=action)

            put_items.append(
                (
                    node_name,
                    size,
                    move_str,
                    moved_piece_type,
                    captured_piece_type,
                    board.is_check(),
                    score,
                )
            )
        self.memory_manager.set_many_boards(to_set)

        while True:
            try:
                self.output_queue.put_many(put_items)
                break
            except Full:
                time.sleep(self.SLEEP)

    def profile_code(self):
        profiler = Profiler()
        profiler.start()

        def before_exit(*args):
            """"""
            # print(f"call count: {self.call_count}")
            profiler.stop()
            profiler.print(show_all=True)
            # self.terminate()
            exit(0)

        signal(SIGTERM, before_exit)
