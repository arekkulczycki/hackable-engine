# -*- coding: utf-8 -*-
"""
Controls all engine flows and communication to outside world.
"""
from typing import Optional

import chess

from arek_chess.main.search_tree_manager import SearchTreeManager
from arek_chess.utils.messaging import Queue
from arek_chess.workers.eval_worker import EvalWorker
from arek_chess.workers.selector_worker import SelectorWorker

CPU_CORES = 8

DEPTH = 6


class Controller:
    """
    Controls all engine flows and communication to outside world.
    """

    def __init__(self):
        """"""

        self.eval_queue = Queue("eval")
        self.candidates_queue = Queue("candidate")
        self.selector_queue = Queue("selector")

        self.board: chess.Board
        self.tree_manager = SearchTreeManager(self.eval_queue, self.candidates_queue, DEPTH)

        self.child_processes = []
        for _ in range(
            CPU_CORES - 2
        ):  # one process required for the tree search and one for the selector worker
            evaluator = EvalWorker(self.eval_queue, self.selector_queue)
            self.child_processes.append(evaluator)

        selector = SelectorWorker(self.selector_queue, self.candidates_queue)
        self.child_processes.append(selector)

    def boot_up(self, fen: Optional[str] = None) -> None:
        """"""

        self.board = chess.Board(fen) if fen else chess.Board()

        self.tree_manager.set_root(fen)

        for process in self.child_processes:
            process.start()

    def make_move(self) -> None:
        """"""

        move = self.tree_manager.search()

        # print(move)

        self.board.push(chess.Move.from_uci(move))

        self.tree_manager.set_root(self.board.fen())

    def get_best_move(self, fen: str, turn: Optional[bool] = None) -> str:
        """"""

        self.tree_manager.set_root(fen, turn)

        return self.tree_manager.search()

    def tear_down(self) -> None:
        """"""

        for process in self.child_processes:
            process.terminate()
