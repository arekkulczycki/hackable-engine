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
from arek_chess.criteria.evaluation.base_eval import ActionType
from arek_chess.game_tree.node import Node
from arek_chess.workers.distributor_worker import DistributorWorker
from arek_chess.workers.eval_worker import EvalWorker
from arek_chess.workers.search_worker import SearchWorker

CPU_CORES = 8


class Controller:
    """
    Controls all engine flows and communication to outside world.
    """

    distributor_queue: QueueManager
    eval_queue: QueueManager
    selector_queue: QueueManager
    control_queue: QueueManager
    board: Board

    def __init__(
        self,
        printing: Union[Print, int],
        tree_params: str = "",
        search_limit: Optional[int] = None,
        model_version: Optional[str] = None,
        timeout: Optional[float] = None,
        memory_action: bool = False,
        in_thread: bool = False,
    ):
        """"""

        self.printing = printing
        self.tree_params = tree_params
        self.search_limit = search_limit
        self.timeout = timeout
        self.original_timeout = timeout
        self.in_thread = in_thread

        self.queue_throttle = (CPU_CORES - 2) * (
            search_limit if search_limit < 12 else (search_limit * 3)
        )

        self.create_queues()

        self.initial_fen: Optional[str] = None
        self.child_processes: List = []

        self.model_version = model_version
        self.memory_action = memory_action

    def boot_up(self, fen: Optional[str] = None) -> None:
        """"""

        if fen:
            self.initial_fen = fen
            self.board = Board(fen)
        elif self.initial_fen:
            self.board = Board(self.initial_fen)
        else:
            self.board = Board()
            self.initial_fen = self.board.fen()

        self.start_child_processes()
        self._setup_search_worker()

    def start_child_processes(self) -> None:
        """"""

        num_eval_workers = max(
            1, CPU_CORES - 2
        )  # one process required for the tree search and one for the distributor worker

        from arek_chess.training.envs.square_control_env import SquareControlEnv

        for _ in range(num_eval_workers):
            evaluator = EvalWorker(
                self.eval_queue,
                self.selector_queue,
                (self.queue_throttle // num_eval_workers),
                memory_action=self.memory_action,
                env=self.model_version and SquareControlEnv(self),
                model_version=self.model_version,
            )
            self.child_processes.append(evaluator)

        distributor = DistributorWorker(
            self.distributor_queue,
            self.eval_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
        )
        self.child_processes.append(distributor)

        for process in self.child_processes:
            process.start()

    def _restart_search_worker(self):
        self.search_worker.stop()  # sends a signal for the thread to finish
        if self.search_worker._started.is_set():
            self.search_worker.join(0.5)
        sleep(0.5)
        self.clear_queues()
        self._setup_search_worker()

    def _setup_search_worker(self):
        self.search_worker = SearchWorker(
            self.distributor_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
            self.printing,
            self.tree_params,
            self.search_limit,
        )
        self.search_worker.set_root(self.board)  # TODO: set previously calculated tree as new root

    def make_move(
        self,
        memory_action: Optional[ActionType] = None,
    ) -> None:
        """"""

        if memory_action is not None:
            MemoryManager().set_action(memory_action, len(memory_action))

        self._setup_search_worker()

        fails = 0
        while True:
            if fails > 5:
                raise RuntimeError
            elif fails >= 3:
                print("restarting all workers...")
                self.restart(fen=self.board.fen())
                sleep(3)
            elif fails > 0:
                print("restarting search worker...")
                self._restart_search_worker()
            try:
                move = self._search()
                # self.timeout = self.original_timeout
                break
            except SearchFailed as e:
                # self.timeout += 1
                if fails >= 2:
                    traceback.print_exc()
                else:
                    print(f"search failed: {e}\nstarting over...")
                fails += 1

        self.release_memory()
        self.clear_queues()

        self.board.push(move)

    def make_move_and_get_root_node(self) -> Node:
        self.make_move()
        return self.search_worker.root

    def _search(self) -> Move:
        # TODO: current thread vs new thread vs new process???
        if self.in_thread:
            self.search_worker.start()
            move = self.search_worker.join(self.timeout)
        else:
            move = self.search_worker._search()
        if not move:
            raise SearchFailed

        return Move.from_uci(move)

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

        print(f"game over, result: {self.board.result()}, {termination}")

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

    def restart(self, fen: Optional[str] = None) -> None:
        """"""

        self.release_memory()
        self.restart_child_processes()

        self.board = Board(fen or self.initial_fen)
        # self.search_manager.set_root(self.board)

        # TODO: restart the action to default
        # MemoryManager().set_action(self.evaluator_class.DEFAULT_ACTION, self.evaluator_class.DEFAULT_ACTION)

    def restart_child_processes(self) -> None:
        """"""

        self.stop_child_processes()
        self.recreate_queues()
        self.start_child_processes()
        self._restart_search_worker()

        # print("processes restarted...")

    def create_queues(self):
        """"""

        self.distributor_queue: QueueManager = QueueManager("distributor")
        self.eval_queue: QueueManager = QueueManager("evaluator")
        self.selector_queue: QueueManager = QueueManager("selector")
        self.control_queue: QueueManager = QueueManager("control")

    def recreate_queues(self):
        """"""

        self.clear_queues()

        self.distributor_queue.close()
        self.eval_queue.close()
        self.selector_queue.close()
        self.control_queue.close()

        self.create_queues()

    def clear_queues(self):
        """"""

        for queue in [
            self.eval_queue,
            self.control_queue,
            self.selector_queue,
        ]:
            items = queue.get_many_blocking(0.01, 10)
            while items:
                items = queue.get_many_blocking(0.01, 10)

    def tear_down(self) -> None:
        """"""

        print("closing engine")

        self.stop_child_processes()

        self.release_memory()

    def stop_child_processes(self) -> None:
        """"""

        for process in self.child_processes:
            process.terminate()
            process.join(1)

        self.child_processes.clear()

    @staticmethod
    def release_memory() -> None:
        """"""

        MemoryManager().clean()
