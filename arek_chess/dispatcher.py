# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from random import randint

from anytree import Node

from arek_chess.board.board import Board
from arek_chess.messaging import Queue
from arek_chess.models.stoppable_thread import StoppableThread
from arek_chess.workers.eval_worker import EvalWorker
from arek_chess.workers.selection_worker import SelectionWorker


class Dispatcher(StoppableThread):
    """
    Class_docstring
    """

    SLEEP = 0.01

    def __init__(self, node_queue: Queue, candidates_queue: Queue, name=None):
        self.node_queue = node_queue
        self.candidates_queue = candidates_queue

        self.eval_queue = Queue(name="eval")
        self.selector_queue = Queue(name="selector")

        self.eval = EvalWorker(self.eval_queue, self.selector_queue)
        self.selector = SelectionWorker(self.selector_queue, self.candidates_queue)

        super().__init__(name=name)

    def run(self):
        """

        :return:
        """

        self.eval.start()
        self.selector.start()

        while self.running:
            node_item = self.node_queue.get()
            if node_item:
                node_name, node_fen = node_item
                board = Board(node_fen)

                moves = list(board.legal_moves)
                moves_n = len(moves)
                for move in moves:
                    self.eval_queue.put((node_name, moves_n, move, node_fen))

            else:
                time.sleep(self.SLEEP)
        else:
            self.eval.terminate()
            self.selector.terminate()
