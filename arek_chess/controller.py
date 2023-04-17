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
from arek_chess.workers.distributor_worker import DistributorWorker
from arek_chess.workers.eval_worker import EvalWorker
from stable_baselines3 import PPO

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
        timeout: Optional[int] = None
    ):
        """"""

        self.printing = printing
        self.tree_params = tree_params
        self.search_limit = search_limit
        self.timeout = timeout

        self.queue_throttle = (CPU_CORES - 2) * (search_limit if search_limit < 12 else (search_limit * 3))

        self.create_queues()

        self.initial_fen: Optional[str] = None
        self.child_processes: List = []
        self.model = None

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

        # self.search_manager.set_root(self.board)

        # self.setup_model()

        self.start_child_processes(action)

    def setup_model(self):
        """"""

        from arek_chess.training.envs.square_control_env import SquareControlEnv
        self.env = SquareControlEnv(self)
        self.model = PPO.load(
            f"./chess.67",
            env=self.env,
        )

    def start_child_processes(
        self, action: Optional[BaseEval.ActionType] = None
    ) -> None:
        """"""

        num_eval_workers = CPU_CORES - 2  # one process required for the tree search and one for the distributor worker

        for _ in range(num_eval_workers):
            evaluator = EvalWorker(
                self.eval_queue,
                self.selector_queue,
                (self.queue_throttle // num_eval_workers),
                action is None and self.model is None,
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

    def _start_search_worker(self):
        self.search_worker = SearchWorker(
            self.distributor_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
            self.printing,
            self.tree_params,
            self.search_limit,
        )
        self.search_worker.set_root(self.board)

    def make_move(self, action: Optional[BaseEval.ActionType] = None, evaluator_class: Optional[str] = None) -> None:
        """"""

        self._start_search_worker()

        if self.model is not None:
            obs = self.env.observation()
            action = self.model.predict(obs)[0]
            print(action)
            MemoryManager().set_action(action, len(action))
        elif action is not None:
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
                # move = self.search_manager.search()
                move = self.search()
                break
            except SearchFailed as e:
                if fails >= 2:
                    traceback.print_exc()
                else:
                    print(f"search failed: {e}\nstarting over...")
                fails += 1

                # self.search_manager.set_root(Board(self.board.fen()))
                self.search_worker.join(0.5)
                sleep(0.5)
                self._start_search_worker()

        self.release_memory()

        self.board.push(move)

        # self.search_manager.set_root(Board(self.board.fen()))

    def search(self) -> Move:
        # TODO: current thread vs new thread vs new process???
        self.search_worker.start()
        move = self.search_worker.join(self.timeout)
        # move = self.search_worker._search()
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
        # self.search_manager.set_root(self.board)

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

        self.distributor_queue: QueueManager = QueueManager("distributor")
        self.eval_queue: QueueManager = QueueManager("evaluator")
        self.selector_queue: QueueManager = QueueManager("selector")
        self.control_queue: QueueManager = QueueManager("control")

        # self.search_manager = SearchManager(
        #     self.distributor_queue,
        #     self.selector_queue,
        #     self.control_queue,
        #     self.queue_throttle,
        #     self.printing,
        #     self.tree_params,
        #     self.search_limit,
        # )

    def recreate_queues(self):
        """"""

        self.clear_queues()
        # del self.search_manager
        del self.distributor_queue
        del self.eval_queue
        del self.selector_queue
        del self.control_queue

        self.create_queues()

    def clear_queues(self):
        """"""

        for queue in [self.distributor_queue, self.eval_queue, self.selector_queue, self.control_queue]:
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
            process.join(1)

        self.child_processes.clear()

    @staticmethod
    def release_memory() -> None:
        """"""

        MemoryManager().clean()
