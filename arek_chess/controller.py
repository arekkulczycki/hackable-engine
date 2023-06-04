# -*- coding: utf-8 -*-
import traceback
from time import sleep
from typing import Optional, List, Union
from weakref import WeakValueDictionary

from chess import Move, Termination

from arek_chess.board.board import Board
from arek_chess.common.constants import Print, ROOT_NODE_NAME
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory.manager import MemoryManager
from arek_chess.common.queue.items.control_item import ControlItem
from arek_chess.common.queue.items.distributor_item import DistributorItem
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager
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
    search_worker: SearchWorker

    def __init__(
        self,
        printing: Union[Print, int],
        tree_params: str = "",
        search_limit: Optional[int] = None,
        model_version: Optional[str] = None,
        is_training_run: bool = False,
        in_thread: bool = False,
        timeout: Optional[float] = None,
    ):
        """"""

        self.printing = printing
        self.tree_params = tree_params
        self.search_limit = search_limit or 16
        self.original_timeout = timeout
        self.in_thread = in_thread
        self.timeout = timeout

        self.queue_throttle = (CPU_CORES - 2) * (
            self.search_limit if self.search_limit < 12 else (self.search_limit * 3)
        )

        self.create_queues()
        self.search_worker = SearchWorker(
            self.distributor_queue,
            self.selector_queue,
            self.control_queue,
            self.queue_throttle,
            self.printing,
            self.tree_params,
            self.search_limit,
        )

        self.initial_fen: Optional[str] = None
        self.child_processes: List = []

        self.model_version = model_version
        self.is_training_run = is_training_run

        # cache search results for subsequent moves
        self.initial_root_color: bool = True
        self.last_white_root: Optional[Node] = None
        self.last_black_root: Optional[Node] = None
        self.last_white_nodes_dict: WeakValueDictionary[
            str, Node
        ] = WeakValueDictionary({})
        self.last_black_nodes_dict: WeakValueDictionary[
            str, Node
        ] = WeakValueDictionary({})

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

        self.initial_root_color = self.board.turn
        self.start_child_processes()

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
                is_training_run=self.is_training_run,
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

    def _restart_search_worker(self, run_iteration: int = 0):
        if self.search_worker and self.in_thread:
            self.search_worker.stop()  # sends a signal for the thread to finish
            if self.search_worker._started.is_set():
                self.search_worker.join(0.5)

        sleep(0.5)
        # self.clear_queues()
        self._setup_search_worker(run_iteration)

    def _setup_search_worker(self, run_iteration: int = 0, reuse: bool = False):
        """"""

        # if reuse and self.search_worker.root is None:
        #     raise ValueError("Cannot reuse root when no root was set yet")
        if self.search_worker.root is None:
            reuse = False

        # cache previous run tree if there was a move
        next_root = None
        next_nodes_dict = None
        if reuse:
            if self.is_training_run:
                # make sure it re-uses old tree every full move, not sharing tree with the opponent
                if self.search_worker.root.color and self.last_black_root:
                    next_root = self._get_next_grandroot(self.last_black_root)
                    if next_root:
                        next_nodes_dict = self._remap_nodes_dict(
                            self.last_black_nodes_dict, next_root.name
                        )
                elif not self.search_worker.root.color and self.last_white_root:
                    next_root = self._get_next_grandroot(self.last_white_root)
                    if next_root:
                        next_nodes_dict = self._remap_nodes_dict(
                            self.last_white_nodes_dict, next_root.name
                        )

            else:
                next_root = self._get_next_root()
                next_nodes_dict = self._remap_nodes_dict(
                    self.search_worker.nodes_dict, next_root.name
                )

        if reuse and next_root:
            self.search_worker.restart(
                self.board.copy(),
                next_root,
                next_nodes_dict,
                run_iteration,
            )
        else:
            self.search_worker.restart(self.board.copy(), run_iteration=run_iteration)

        if reuse and self.is_training_run:
            if self.search_worker.root.color:
                self.last_white_root = self.search_worker.root
                self.last_white_nodes_dict = self.search_worker.nodes_dict
            else:
                self.last_black_root = self.search_worker.root
                self.last_black_nodes_dict = self.search_worker.nodes_dict

    def _get_next_root(self) -> Node:
        chosen_move = self.board.move_stack[-1].uci()

        chosen_child: Optional[Node] = None
        for child in self.search_worker.root.children:
            if child.move == chosen_move:
                chosen_child = child
                break

        if chosen_child is None:
            raise ValueError(
                f"Next root: Could not recognize move played: {chosen_move}. "
                f"Children: {' - '.join([str(child) for child in self.search_worker.root.children])}"
            )

        return chosen_child

    def _get_next_grandroot(self, node: Node) -> Optional[Node]:
        if len(self.board.move_stack) < 2:
            return None

        chosen_child_move = self.board.move_stack[-2].uci()
        chosen_grandchild_move = self.board.move_stack[-1].uci()

        chosen_child: Optional[Node] = None
        for child in node.children:
            if child.move == chosen_child_move:
                chosen_child = child
                break

        if chosen_child is None:
            raise ValueError(
                f"Next grand-root: Could not recognize move played: {chosen_child_move}. "
                f"Children: {' - '.join([str(child) for child in node.children])}"
            )

        chosen_grandchild: Optional[Node] = None
        for child in chosen_child.children:
            if child.move == chosen_grandchild_move:
                chosen_grandchild = child
                break

        return chosen_grandchild

    def _remap_nodes_dict(
        self, nodes_dict: WeakValueDictionary, root_name: str
    ) -> WeakValueDictionary:
        """
        Clean hashmap of discarded moves and rename remaining keys.
        """

        for key in list(nodes_dict.keys()):
            if key.startswith(root_name):  # TODO: not needed for the weakref stuff ??
                new_key = key.replace(root_name, ROOT_NODE_NAME)
                nodes_dict[new_key] = nodes_dict.pop(key)
            else:
                del nodes_dict[key]  # TODO: not needed for the weakref stuff ??

        return nodes_dict

    def make_move(self, memory_action: Optional[ActionType] = None) -> None:
        """"""

        if self.board.is_checkmate():
            print("asked for a move in checkmate position")
            print(self.get_pgn())
            raise ValueError("asked for a move in checkmate position")

        self._setup_search_worker(
            reuse=True
        )  # TODO: set True when branch reusing works

        if memory_action is not None:
            MemoryManager().set_action(memory_action, len(memory_action))

        fails = 0
        while True:
            if fails > 3:
                raise RuntimeError
            elif fails > 1:
                print("restarting all workers...")
                sleep(3 * fails)
                self.restart(fen=self.board.fen())
                # self.restart()
                sleep(3 * fails)
                self.restart_child_processes()
            elif fails > 0:
                print("restarting search worker...")
                # self.release_memory(silent=self.printing in [Print.MOVE, Print.MOVE])
                self.restart(fen=self.board.fen())

            try:
                move = self._search()
                # self.timeout = self.original_timeout
                break
            except SearchFailed as e:
                print(self.initial_fen)
                print(self.get_pgn())
                fails += 1
            except TimeoutError as e:
                fails += 1
                print(f"timed out in search: {e}")
                sleep(fails)

        try:
            self.board.push(Move.from_uci(move))
        except AssertionError:
            # move illegal ???
            print("found illegal move, restarting...")
            self.restart(self.initial_fen)
            self.make_move()

        self.clear_queues()

    def get_move(self) -> str:
        # TODO: rework to be an internal part of the make_move
        self._setup_search_worker()
        return self._search()

    def _search(self) -> str:
        # TODO: current thread vs new thread vs new process???
        if self.in_thread:
            self.search_worker.start()
            move = self.search_worker.join(self.timeout)
            if self.search_worker.is_alive():
                raise TimeoutError
        else:
            move = self.search_worker._search()
        if not move:
            raise SearchFailed

        return move

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

        fen = self.board.fen()
        board = Board(self.initial_fen)
        try:
            pgn = board.variation_san(self.board.move_stack)
        except ValueError:
            print("ValueError getting pgn for")
            print(board.fen())
            return ".".join([move.uci() for move in self.board.move_stack])

        return f'[FEN "{fen}"]\n\n{pgn}'

    def restart(self, fen: Optional[str] = None) -> None:
        """"""

        self.board = Board(fen or self.initial_fen)

        self.initial_root_color: bool = self.board.turn
        self.last_white_root: Optional[Node] = None
        self.last_black_root: Optional[Node] = None
        self.last_white_nodes_dict: WeakValueDictionary[
            str, Node
        ] = WeakValueDictionary({})
        self.last_black_nodes_dict: WeakValueDictionary[
            str, Node
        ] = WeakValueDictionary({})

        # self.release_memory(
        #     except_prefix=next_root.name,
        #     silent=self.printing in [Print.MOVE, Print.MOVE],
        # )
        # self.release_memory(silent=self.printing in [Print.NOTHING, Print.MOVE])
        # self.restart_child_processes()

        self._restart_search_worker()

        # MemoryManager().set_action(self.evaluator_class.DEFAULT_ACTION, self.evaluator_class.DEFAULT_ACTION)

    def restart_child_processes(self) -> None:
        """"""

        print("stopping child processes")
        self.stop_child_processes()

        print("recreating queues")
        self.recreate_queues()

        print("starting child processes")
        self.start_child_processes()
        # self._restart_search_worker()

        # print("processes restarted...")

    def create_queues(self):
        """"""

        self.distributor_queue: QueueManager = QueueManager(
            "distributor", loader=DistributorItem.loads, dumper=DistributorItem.dumps
        )
        self.eval_queue: QueueManager = QueueManager(
            "evaluator", loader=EvalItem.loads, dumper=EvalItem.dumps
        )
        self.selector_queue: QueueManager = QueueManager(
            "selector", loader=SelectorItem.loads, dumper=SelectorItem.dumps
        )
        self.control_queue: QueueManager = QueueManager(
            "control", loader=ControlItem.loads, dumper=ControlItem.dumps
        )

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

        # for queue in [
        #     self.control_queue,
        #     self.selector_queue,
        #     self.distributor_queue,
        #     self.eval_queue,
        # ]:
        #     queue.get_many(10, SLEEP)
        #     while queue.get_many(100, SLEEP):
        #         print(f"cleaned leftover items from {queue.name}")

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
    def release_memory(except_prefix: str = "", *, silent: bool = False) -> None:
        """"""

        MemoryManager().clean(except_prefix, silent)
