"""
Controls all engine flows and communication to outside world.
"""

from typing import Optional

import chess

from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.constants import Print
from arek_chess.main.search_manager import SearchManager
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager
from arek_chess.workers.dispatcher_worker import DispatcherWorker
from arek_chess.workers.eval_worker import EvalWorker

CPU_CORES = 8


class Controller:
    """
    Controls all engine flows and communication to outside world.
    """

    def __init__(self, printing: Print, tree_params: str):
        """"""

        self.dispatcher_queue: QueueManager = QueueManager("dispatcher")
        self.eval_queue: QueueManager = QueueManager("evaluator")
        self.selector_queue: QueueManager = QueueManager("selector")

        self.board: chess.Board
        self.search_manager = SearchManager(
            self.dispatcher_queue,
            self.eval_queue,
            self.selector_queue,
            printing,
            tree_params,
        )

        self.initial_fen: Optional[str] = None

    def boot_up(
        self, fen: Optional[str] = None, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        if fen:
            self.initial_fen = fen
            self.board = chess.Board(fen)
        elif self.initial_fen:
            self.board = chess.Board(self.initial_fen)
        else:
            self.board = chess.Board()
            self.initial_fen = self.board.fen()

        self.search_manager.set_root(self.initial_fen)
        if action:
            MemoryManager().set_action(action, len(action))

        self.child_processes = []
        for _ in range(
            CPU_CORES - 2
        ):  # one process required for the tree search and one for the dispatcher worker
            evaluator = EvalWorker(self.eval_queue, self.selector_queue, action is None)
            self.child_processes.append(evaluator)

        dispatcher = DispatcherWorker(self.dispatcher_queue, self.eval_queue)
        self.child_processes.append(dispatcher)

        for process in self.child_processes:
            process.start()

    def make_move(self) -> None:
        """"""

        move = self.search_manager.search()
        self.release_memory()

        # print(move)

        self.board.push(chess.Move.from_uci(move))

        self.search_manager.set_root(self.board.fen())

    def play(self) -> None:
        """"""

        while not self.board.is_game_over():
            self.make_move()
            print(self.board.fen())

        print(
            f"game over, result: {self.board.result()}, {self.board.outcome().termination}"
        )

    def get_best_move(self, fen: str, turn: Optional[bool] = None) -> str:
        """"""

        self.search_manager.set_root(fen, turn)

        return self.search_manager.search()

    def restart(self, action: Optional[BaseEval.ActionType] = None) -> None:
        """"""

        self.board = chess.Board(self.initial_fen)
        self.search_manager.set_root(self.initial_fen)
        if action:
            MemoryManager().set_action(action, len(action))

    def tear_down(self) -> None:
        """"""

        for process in self.child_processes:
            process.terminate()

        self.release_memory()

    @staticmethod
    def release_memory() -> None:
        """"""

        MemoryManager().clean()
