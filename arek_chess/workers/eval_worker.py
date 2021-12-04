# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from typing import Tuple

import numpy

from arek_chess.board.board import Board
from arek_chess.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker

DEFAULT_ACTION = numpy.array([100.0, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float32)


class EvalWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.001

    def __init__(self, eval_queue: Queue, selector_queue: Queue):
        super().__init__()

        self.eval_queue = eval_queue
        self.selector_queue = selector_queue

    def run(self):
        """

        :return:
        """

        while True:
            eval_item = self.eval_queue.get()
            if eval_item:
                group, size, move, fen = eval_item
                score, new_fen = self.get_score(move, fen)

                self.selector_queue.put((group, size, move, new_fen, score))
            else:
                time.sleep(self.SLEEP)

    def get_score(self, move, fen) -> Tuple[float, str]:
        """

        :return:
        """

        board = Board(fen)
        board.push(move)

        return board.get_score(DEFAULT_ACTION, board.get_moving_piece_type(move)), board.fen()
