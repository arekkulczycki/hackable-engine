# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM
from typing import List

from pyinstrument import Profiler

from arek_chess.criteria.evaluation.arek_eval import ArekEval
from arek_chess.utils.memory_manager import (
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

        self.call_count = 0
        #
        # self.profile_code()

        self.evaluator = ArekEval()

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

                self.call_count += 1

                # benchmark max perf with random generation
                # score = uniform(-10, 10)
                score = self.evaluator.get_score(
                    self.action, node_name, not turn_after, move, captured_piece_type
                )

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
