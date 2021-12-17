# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from typing import List

from arek_chess.board.board import Board
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.messaging import Queue
from arek_chess.utils.stoppable_thread import StoppableThread
from arek_chess.workers.eval_worker import EvalWorker
from arek_chess.workers.selection_worker import SelectionWorker

CPU_CORES = 8


class Dispatcher(StoppableThread):
    """
    Class_docstring
    """

    SLEEP = 0.001

    def __init__(
        self,
        node_queue: Queue,
        candidates_queue: Queue,
        action: List[float],
    ):
        self.node_queue = node_queue
        self.candidates_queue = candidates_queue

        self.eval_queue = Queue(name="eval")
        self.selector_queue = Queue(name="selector")

        self.child_processes = []
        for _ in range(CPU_CORES - 2):
            evaluator = EvalWorker(self.eval_queue, self.selector_queue, action)
            evaluator.start()
            self.child_processes.append(evaluator)

        selector = SelectionWorker(
            self.selector_queue, self.candidates_queue
        )
        selector.start()
        self.child_processes.append(selector)

        super().__init__()

    def run(self):
        """

        :return:
        """

        # profiler = Profiler()
        # profiler.start()

        while self.running:
            node_item = self.node_queue.get()
            if node_item:
                node_name, node_turn = node_item

                board = MemoryManager.get_node_board(node_name)
                board.turn = node_turn

                moves = [move for move in board.legal_moves]

                if moves:
                    self.create_node_params_cache(board, node_name)
                else:  # is checkmate
                    self.candidates_queue.put([{
                        "node_name": f"{node_name}.0",
                        "move": "checkmate",
                        "turn": not node_turn,
                        "score": 1000000 if node_turn else -1000000,
                        "is_capture": False,
                    }])

                moves_n = len(moves)

                for move in moves:
                    captured_piece_type = board.get_captured_piece_type(move)

                    self.eval_queue.put(
                        (
                            node_name,
                            moves_n,
                            move.uci(),
                            not node_turn,
                            captured_piece_type,
                        )
                    )

            else:
                time.sleep(self.SLEEP)
        else:
            # profiler.stop()
            # profiler.print(show_all=True)
            for p in self.child_processes:
                p.terminate()

    def create_node_params_cache(self, board: Board, node_name) -> None:
        white_params = [
            *board.get_material_and_safety(True),
            *board.get_total_mobility(True),
        ]
        black_params = [
            *board.get_material_and_safety(False),
            *board.get_total_mobility(False),
        ]

        MemoryManager.set_node_params(node_name, *[white - black for white, black in zip(white_params, black_params)])
