"""
Controls all engine flows and communication to outside world.
"""

import traceback
from time import sleep
from typing import Optional, List, Union

from chess import Move, Termination

from arek_chess.board.board import Board
from arek_chess.common.constants import Print
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.queue_manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.game_tree.search_manager import SearchManager
from arek_chess.workers.dispatcher_worker import DispatcherWorker
from arek_chess.workers.eval_worker import EvalWorker

CPU_CORES = 8


class Controller:
    """
    Controls all engine flows and communication to outside world.
    """

    dispatcher_queue: QueueManager
    eval_queue: QueueManager
    selector_queue: QueueManager
    control_queue: QueueManager
    board: Board

    def __init__(
        self,
        printing: Union[Print, int],
        tree_params: str = "",
        search_limit: Optional[int] = None,
    ):
        """"""

        self.printing = printing
        self.tree_params = tree_params
        self.search_limit = search_limit

        self.queue_throttle = (CPU_CORES - 2) * search_limit * 2

        self.create_queues()

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
                (self.queue_throttle // CPU_CORES - 2),
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

        fails = 0
        while True:
            if fails == 3:
                self.restart(fen=self.board.fen())
                sleep(0.5)
            elif fails > 4:
                raise RuntimeError
            try:
                move = self.search_manager.search()
                break
            except SearchFailed as e:
                if fails >= 2:
                    traceback.print_exc()
                else:
                    print(f"search failed: {e}\nstarting over...")
                fails += 1

                self.search_manager.set_root(Board(self.board.fen()))
                sleep(0.5)

        self.release_memory()

        if not move:
            raise RuntimeError
        chess_move = Move.from_uci(move)
        self.board.push(chess_move)

        self.search_manager.set_root(Board(self.board.fen()))

    def search(self) -> Move:
        move = self.search_manager.search()
        if not move:
            raise RuntimeError
        chess_move = Move.from_uci(move)
        # self.board.push(chess_move)
        self.release_memory()
        return chess_move

    def play(self) -> None:
        """"""

        while not self.board.is_game_over() and self.board.outcome() is None:
            self.make_move()
            print(self.board.fen())
            sleep(0.05)

        self.tear_down()

        outcome = self.board.outcome()

        termination = ""
        for key, value in Termination.__dict__.items():
            if value == outcome.termination:
                termination = key

        print(
            f"game over, result: {self.board.result()}, {termination}"
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

    def restart(self, fen: Optional[str] = None, action: Optional[BaseEval.ActionType] = None) -> None:
        """"""

        print("restarting engine")

        self.release_memory()
        self.restart_child_processes(action)

        self.board = Board(fen or self.initial_fen)
        self.search_manager.set_root(self.board)

        if action is not None:
            MemoryManager().set_action(action, len(action))

    def restart_child_processes(
        self, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        self.stop_child_processes()
        self.recreate_queues()
        self.start_child_processes(action)

        # print("processes restarted...")

    def create_queues(self):
        """"""

        self.dispatcher_queue: QueueManager = QueueManager("dispatcher")
        self.eval_queue: QueueManager = QueueManager("evaluator")
        self.selector_queue: QueueManager = QueueManager("selector")
        self.control_queue: QueueManager = QueueManager("control")

        self.search_manager = SearchManager(
            self.dispatcher_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
            self.printing,
            self.tree_params,
            self.search_limit,
        )

    def recreate_queues(self):
        """"""

        self.clear_queues()
        del self.search_manager
        del self.dispatcher_queue
        del self.eval_queue
        del self.selector_queue
        del self.control_queue

        self.create_queues()

    def clear_queues(self):
        """"""

        for queue in [self.dispatcher_queue, self.eval_queue, self.selector_queue, self.control_queue]:
            while not queue.empty():
                queue.get_many()

    def tear_down(self) -> None:
        """"""

        print("closing engine")

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
