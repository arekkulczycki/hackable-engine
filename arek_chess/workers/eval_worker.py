# -*- coding: utf-8 -*-
"""
Worker that performs evaluation on nodes picked up from queue.
"""

import sys
from signal import signal, SIGTERM
from time import sleep
from typing import Tuple

from faster_fifo import Full
from pyinstrument import Profiler

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.legacy_eval import LegacyEval
from arek_chess.constants import INF
from arek_chess.utils.queue_manager import QueueManager
from arek_chess.workers.base_worker import BaseWorker


class EvalWorker(BaseWorker):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    SLEEP = 0.0001

    def __init__(
        self,
        input_queue: QueueManager,
        output_queue: QueueManager,
        constant_action: bool = False,
    ):
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

        self.max_items_at_once = 128

    def _run(self):
        """

        :return:
        """

        self.setup()
        memory_manager = self.memory_manager
        input_queue = self.input_queue
        max_items_at_once = self.max_items_at_once
        eval_items = self.eval_items
        _sleep = self.SLEEP

        while True:
            items_to_eval = input_queue.get_many(max_items_at_once)
            if items_to_eval:
                eval_items(items_to_eval, memory_manager)
            else:
                sleep(_sleep)

    def eval_items(self, eval_items, memory_manager):
        """"""

        names = [item[0] for item in eval_items]  # generators are slower :|
        boards = memory_manager.get_many_boards(names)

        queue_items = [
            self.collect_item(board, node_name, move_str)
            for (node_name, move_str), board in zip(eval_items, boards)
        ]

        # self.output_queue.put_many(put_items)
        while True:
            try:
                self.output_queue.put_many(queue_items)
                break
            except Full:
                sleep(self.SLEEP)

    def collect_item(self, board: Board, node_name: str, move_str: str) -> Tuple:
        """"""

        # self.call_count += 1

        board, captured_piece_type, moved_piece_type = self.get_board_data(
            board, node_name, move_str
        )  # board after the move

        # to_set.append((f"{node_name}.{move_str}", board))

        outcome = board.simple_outcome()
        if outcome is not None:
            winner = outcome.winner
            score = 0 if winner is None else INF if winner else -INF

        else:
            action = (
                None
                if self.constant_action
                else self.get_action(self.evaluator.ACTION_SIZE)
            )
            score = self.evaluator.get_score(
                board, move_str, captured_piece_type, action=action
            )

        return (
            node_name,
            move_str,
            moved_piece_type,
            captured_piece_type,
            board.is_check(),
            score,
        )

    def profile_code(self) -> None:
        """"""

        profiler = Profiler()
        profiler.start()

        def before_exit(*_) -> None:
            """"""

            # print(f"call count: {self.call_count}")
            profiler.stop()
            profiler.print(show_all=True)
            # self.terminate()
            sys.exit(0)

        signal(SIGTERM, before_exit)
