# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from multiprocessing.managers import SharedMemoryManager

from pyinstrument import Profiler

from arek_chess.board.board import Board
from arek_chess.common_data_manager import CommonDataManager
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

        self.memory_manager = SharedMemoryManager()
        self.memory_manager.start()

        self.child_processes = []
        for _ in range(6):
            evaluator = EvalWorker(self.eval_queue, self.selector_queue)
            evaluator.start()
            self.child_processes.append(evaluator)

        selector = SelectionWorker(self.selector_queue, self.candidates_queue)
        selector.start()
        self.child_processes.append(selector)

        self.common_data_manager = CommonDataManager()

        super().__init__(name=name)

    def run(self):
        """

        :return:
        """

        # profiler = Profiler()
        # profiler.start()

        while self.running:
            node_item = self.node_queue.get()
            if node_item:
                node_name, node_fen, node_turn = node_item
                board = Board(node_fen)
                board.turn = node_turn

                self.create_node_params_cache(board, node_name)

                moves = [move for move in board.legal_moves]
                moves_n = len(moves)

                for move in moves:
                    captured_piece_type = board.get_captured_piece_type(move)

                    self.eval_queue.put(
                        (node_name, moves_n, move.uci(), node_fen, not node_turn, captured_piece_type)
                    )

            else:
                time.sleep(self.SLEEP)
        else:
            # profiler.stop()
            # profiler.print(show_all=True)
            for p in self.child_processes:
                p.terminate()

    def create_node_params_cache(self, board: Board, node_name) -> None:
        white_params = board.get_material_and_safety(True)
        black_params = board.get_material_and_safety(False)
        # CommonDataManager.create_set_node_memory(
        #     node_name,
        #     *white_params,
        #     *black_params,
        # )

        self.common_data_manager.set_params(node_name, white_params, black_params)
