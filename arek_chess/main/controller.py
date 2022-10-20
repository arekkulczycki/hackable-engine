"""
Controls all engine flows and communication to outside world.
"""
from time import sleep
from typing import Optional, List, Union

from arek_chess.board.board import Move, Board
from arek_chess.common.constants import Print
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.queue_manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.main.search_manager import SearchManager
from arek_chess.workers.dispatcher_worker import DispatcherWorker
from arek_chess.workers.eval_worker import EvalWorker

CPU_CORES = 8


class Controller:
    """
    Controls all engine flows and communication to outside world.
    """

    def __init__(
        self,
        printing: Union[Print, int],
        tree_params: str = "",
        search_limit: Optional[int] = None,
    ):
        """"""

        self.dispatcher_queue: QueueManager = QueueManager("dispatcher")
        self.eval_queue: QueueManager = QueueManager("evaluator")
        self.selector_queue: QueueManager = QueueManager("selector")
        self.control_queue: QueueManager = QueueManager("control")
        self.queue_throttle = 64

        self.board: Board
        self.search_manager = SearchManager(
            self.dispatcher_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
            printing,
            tree_params,
            search_limit,
        )

        self.initial_fen: Optional[str] = None
        self.child_processes: List = []

    def boot_up(
        self, fen: Optional[str] = None, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        if fen:
            self.initial_fen = fen
            self.board = Board(fen)
        elif self.initial_fen:
            self.board = Board(self.initial_fen)
        else:
            self.board = Board()
            self.initial_fen = self.board.fen()

        self.search_manager.set_root(self.board)

        self.start_child_processes(action)

    def start_child_processes(
        self, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        for _ in range(
            CPU_CORES - 2
        ):  # one process required for the tree search and one for the dispatcher worker
            evaluator = EvalWorker(
                self.eval_queue,
                self.selector_queue,
                self.queue_throttle,
                action is None,
            )
            self.child_processes.append(evaluator)

        dispatcher = DispatcherWorker(
            self.dispatcher_queue,
            self.eval_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
        )
        self.child_processes.append(dispatcher)

        for process in self.child_processes:
            process.start()

    def make_move(self, action: Optional[BaseEval.ActionType] = None, evaluator_class: Optional[str] = None) -> None:
        """"""

        if action is not None:
            MemoryManager().set_action(action, len(action))

        # if not self.child_processes:
        #     self.start_child_processes(action)

        while True:
            try:
                move = self.search_manager.search()
                break
            except SearchFailed:
                print("search failed, starting over...")
                self.search_manager.set_root(self.board)
                sleep(0.5)

        # self.tear_down()

        # print(move)

        self.board.push(Move.from_uci(move))

        self.search_manager.set_root(self.board)

    def play(self) -> None:
        """"""

        while not self.board.is_game_over() and self.board.simple_outcome() is None:
            self.make_move()
            print(self.board.fen())
            sleep(0.1)

        outcome = self.board.outcome()
        if outcome is None:
            outcome = self.board.simple_outcome()
        print(
            f"game over, result: {self.board.result()}, {outcome.termination}"
        )

    def get_pgn(self) -> str:
        """"""

        moves = []
        notation = []
        for _ in range(len(self.board.move_stack)):
            move = self.board.pop()
            moves.append(move)
            notation.append(self.board.san(move))

        fen = self.board.fen()
        pgn = self.board.variation_san(reversed(moves))

        for move in reversed(moves):
            self.board.push(move)

        return f'[FEN "{fen}"]\n\n{pgn}'

    def restart(self, action: Optional[BaseEval.ActionType] = None) -> None:
        """"""

        self.release_memory()
        self.restart_child_processes(action)

        self.board = Board(self.initial_fen)
        self.search_manager.set_root(self.board)

        if action is not None:
            MemoryManager().set_action(action, len(action))

    def restart_child_processes(
        self, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        self.stop_child_processes()

        self.start_child_processes(action)

        # print("processes restarted...")

    def tear_down(self) -> None:
        """"""

        self.stop_child_processes()

        self.release_memory()

    def stop_child_processes(self) -> None:
        """"""

        for process in self.child_processes:
            process.terminate()
            process.join(3)

        self.child_processes.clear()

    @staticmethod
    def release_memory() -> None:
        """"""

        MemoryManager().clean()
