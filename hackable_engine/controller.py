# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from multiprocessing import cpu_count
from time import sleep
from typing import List, Optional, TypeVar, Generic, Any, Dict, Type

from hackable_engine.board import GameBoardBase
from hackable_engine.board.chess.chess_board import ChessBoard
from hackable_engine.board.chess.serializers.chess_board_serializer_mixin import (
    CHESS_BOARD_BYTES_NUMBER,
)
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import PROCESS_COUNT
from hackable_engine.common.memory.manager import MemoryManager
from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.common.queue.items.selector_item import SelectorItem
from hackable_engine.common.queue.manager import QueueManager
from hackable_engine.workers.configs.eval_worker_config import EvalWorkerConfig
from hackable_engine.workers.configs.search_tree_cache import (
    SearchTreeCache,
    SideTreeCache,
)
from hackable_engine.workers.configs.worker_locks import WorkerLocks
from hackable_engine.workers.configs.worker_queues import WorkerQueues
from hackable_engine.workers.distributor_worker import DistributorWorker
from hackable_engine.workers.eval_worker import EvalWorker
from hackable_engine.workers.search_worker import SearchWorker

GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)


class Controller(Generic[GameBoardT]):
    """
    Controls engine setup and its behavior while running.
    """

    def __init__(
        self,
        board: GameBoardT,
        model_path: Optional[str] = None,
        is_training_run: bool = False,
    ):
        self.board = board
        self.initial_notation: Optional[str] = board.get_notation()
        self.initial_root_color: bool = not board.turn

        self.memory_manager = MemoryManager()

        self.model_path = model_path
        self.is_training_run = is_training_run
        if is_training_run:  # TODO: implement another way of mock start
            return

        self.worker_queues: WorkerQueues = self._create_queues()
        self.worker_locks: WorkerLocks = WorkerLocks()

        self.search_worker: SearchWorker = SearchWorker(
            self.board, self.worker_locks, self.worker_queues
        )
        self.child_processes: List = []

        self.color_tree_cache = SideTreeCache(
            SearchTreeCache(None, {}), SearchTreeCache(None, {})
        )

    @classmethod
    def configure_for_chess(
        cls,
        board_kwargs: Dict[str, Any],
        model_path: Optional[str] = None,
        is_training: bool = False,
    ) -> Controller[ChessBoard]:
        """Initialize with a ChessBoard and set item sizes that depend on board."""

        board = ChessBoard(**board_kwargs)

        DistributorItem.board_bytes_number = CHESS_BOARD_BYTES_NUMBER
        EvalItem.board_bytes_number = CHESS_BOARD_BYTES_NUMBER
        SelectorItem.board_bytes_number = CHESS_BOARD_BYTES_NUMBER

        return cls(board, model_path, is_training)

    @classmethod
    def configure_for_hex(
        cls,
        board_kwargs: Dict[str, Any],
        model_path: Optional[str] = None,
        is_training: bool = False,
    ) -> Controller[HexBoard]:
        """Initialize with a HexBoard and set item sizes that depend on board."""

        board = HexBoard(**board_kwargs)

        DistributorItem.board_bytes_number = board.board_bytes_number
        EvalItem.board_bytes_number = board.board_bytes_number
        SelectorItem.board_bytes_number = board.board_bytes_number

        return cls(board, model_path, is_training)

    def boot_up(self) -> None:
        """Prepare all the workers of the engine."""

        self._start_child_processes()

    def reset(self, board_kwargs: Dict) -> None:
        """Prepare playing board and caches as if started from scratch."""

        self.reset_board(**board_kwargs)

        self._reset_cache()

        self.setup_search_worker(run_iteration=0)

    def reset_board(self, **board_kwargs):
        """"""

        self._setup_board(self.board.__class__, **board_kwargs)

    @staticmethod
    def _setup_board(board_class: Type[GameBoardT], **kwargs) -> None:
        """
        Initialize the board with an optionally preset position.

        :param position: string representation, for instance fen for chess
        """

        return board_class(**kwargs)

    def _start_child_processes(self) -> None:
        """"""

        num_eval_workers = max(
            1, (PROCESS_COUNT or cpu_count()) - 2
        )  # one process required for the tree search and one for the distributor worker

        # affinity_set = {0, 1, 2}; os.sched_setaffinity(0, affinity_set)
        # TODO: set affinity, force eval worker on weaker cores if exist

        for i in range(num_eval_workers):
            evaluator = EvalWorker(
                self.worker_locks,
                self.worker_queues,
                config=EvalWorkerConfig(
                    worker_number=i + 1,
                    board_class=self.board.__class__,
                    board_size=self.board.size,
                    is_training_run=self.is_training_run,
                    ai_model=self.model_path,
                ),
            )
            self.child_processes.append(evaluator)

        distributor = DistributorWorker(
            self.worker_locks,
            self.worker_queues,
            self.board.__class__,
            self.board.size,
        )
        self.child_processes.append(distributor)

        for process in self.child_processes:
            process.start()

        self._wait_child_processes_ready()

    def _wait_child_processes_ready(self) -> None:
        """"""

        for process in self.child_processes:
            while self.memory_manager.get_int(str(process.pid)) != 1:
                sleep(0.01)

    def setup_search_worker(
        self, run_iteration: int = 0, search_tree: Optional[SearchTreeCache] = None
    ):
        """
        Prepare the tree search initial conditions.

        :param run_iteration: can be used to re-run with same initial conditions and avoid mixups with previous runs
        :param search_tree: a tree from another search that has finished,
            can be reused to dig further or a branch can be taken to support subsequent move search
        """

        if search_tree and self.board.move_stack:
            move_uci: str = self.board.move_stack[-1].uci()

            if move_uci != search_tree.root.move:
                search_tree = self._prepare_search_tree(search_tree, move_uci)

        # cache color-based previous run tree if is in training (separate white and black trees)
        if search_tree and self.is_training_run:
            self._cache_color_based_tree()

        self.search_worker.reset(
            self.board.copy(),
            search_tree,
            run_iteration=run_iteration,
        )

    def _prepare_search_tree(
        self, search_tree: SearchTreeCache, move_uci: str
    ) -> SearchTreeCache:
        """Prepare search tree for the next run based on the one given and the move chosen."""

        if self.is_training_run:
            # make sure it re-uses old tree every full move, not sharing tree with the opponent
            if len(self.board.move_stack) < 2:
                print("insufficient moves played")  # TODO: should raise?
                search_tree = SearchTreeCache(None, {})

            else:
                child_move = self.board.move_stack[-2].uci()
                grandchild_move = self.board.move_stack[-1].uci()
                search_tree = self.color_tree_cache.get_color_based_tree_branch(
                    search_tree, self.board.turn, child_move, grandchild_move
                )

        else:
            search_tree = self.color_tree_cache.get_tree_branch(search_tree, move_uci)

        return search_tree

    def search(self) -> str:
        """Trigger the worker to start exploring."""

        return asyncio.run(self.search_worker.search())

    def _cache_color_based_tree(self) -> None:
        """
        Cache color-based previous run tree and maps to nodes.
        """

        search_tree_cache = SearchTreeCache(
            self.search_worker.root,
            self.search_worker.node_cache.nodes_dict.copy(),
            # self.search_worker.transposition_dict.copy()
        )

        # if white to move, then last move was black, then tree prepared for a next black move
        if self.board.turn:
            self.color_tree_cache.black = search_tree_cache
        else:
            self.color_tree_cache.white = search_tree_cache

    def _reset_cache(self) -> None:
        """"""

        self.color_tree_cache = SideTreeCache(
            SearchTreeCache(None, {}), SearchTreeCache(None, {})
        )

    @staticmethod
    def _create_queues():
        """"""

        return WorkerQueues(
            QueueManager(
                "distributor",
                loader=DistributorItem.loads,
                dumper=DistributorItem.dumps,
            ),
            QueueManager("evaluator", loader=EvalItem.loads, dumper=EvalItem.dumps),
            QueueManager(
                "selector", loader=SelectorItem.loads, dumper=SelectorItem.dumps
            ),
            QueueManager("control", loader=ControlItem.loads, dumper=ControlItem.dumps),
        )

    def tear_down(self) -> None:
        """"""

        if not self.is_training_run:
            self._stop_child_processes()

        self._release_memory()

    def _stop_child_processes(self) -> None:
        """"""

        for process in self.child_processes:
            process.terminate()

        for process in self.child_processes:
            while process.is_alive():
                sleep(0.01)

            self.memory_manager.remove(str(process.pid))
            process.close()

        self.child_processes.clear()

    def _release_memory(self, except_prefix: str = "") -> None:
        """"""

        self.memory_manager.clean(except_prefix)
