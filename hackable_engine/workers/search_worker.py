# -*- coding: utf-8 -*-
import asyncio
import os
from multiprocessing import cpu_count
from random import randint
from time import time
from typing import Generic, List, Optional, Tuple, TypeVar

from numpy import abs as np_abs, float32

from hackable_engine.board import GameBoardBase
from hackable_engine.common.constants import (
    DEBUG,
    DISTRIBUTED,
    INF,
    LOG_INTERVAL,
    PRINT_CANDIDATES,
    PROCESS_COUNT,
    Print,
    QUEUE_THROTTLE,
    ROOT_NODE_NAME,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    TREE_PARAMS,
    WORKER,
    PRINTING,
    SEARCH_LIMIT,
)
from hackable_engine.common.custom_threads import ReturningThread
from hackable_engine.common.exceptions import SearchFailed
from hackable_engine.common.memory.manager import MemoryManager
from hackable_engine.common.profiler_mixin import ProfilerMixin
from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.selector_item import SelectorItem
from hackable_engine.common.queue.manager import QueueManager as QM
from hackable_engine.game_tree.node import Node
from hackable_engine.game_tree.renderer import PrunedTreeRenderer
from hackable_engine.game_tree.traverser import Traverser
from hackable_engine.workers.configs.search_tree_cache import (
    SearchTreeCache,
)
from hackable_engine.workers.configs.search_worker_counters import SearchWorkerCounters
from hackable_engine.workers.configs.search_worker_flags import SearchWorkerFlags
from hackable_engine.workers.configs.worker_locks import WorkerLocks
from hackable_engine.workers.configs.worker_queues import WorkerQueues

GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)


class SearchWorker(ReturningThread, ProfilerMixin, Generic[GameBoardT]):
    """
    Handles the in-memory game tree and controls all tree expansion logic and progress logging.
    """

    limit: int

    def __init__(
        self,
        board: GameBoardT,
        locks: WorkerLocks,
        queues: WorkerQueues,
    ):
        super().__init__()

        self.memory_manager = MemoryManager()

        self.board: GameBoardT = board
        self.locks: WorkerLocks = locks
        self.queues: WorkerQueues = queues

        self.counters: SearchWorkerCounters = SearchWorkerCounters()
        self.flags: SearchWorkerFlags = SearchWorkerFlags()

        root = self._create_root()
        self.node_cache = SearchTreeCache(root, {ROOT_NODE_NAME: root})

        self.traverser: Traverser = Traverser(self.node_cache)

        self.run_id = self._get_run_id()
        self._reset_counters()

    def _reset_counters(self) -> None:
        """"""

        self.counters = SearchWorkerCounters()
        self.flags = SearchWorkerFlags()

        with self.locks.status_lock:
            self.memory_manager.set_int(STATUS, Status.CLOSED)
            self.memory_manager.set_int(DEBUG, 0)

        with self.locks.finish_lock:
            for i in range((PROCESS_COUNT or cpu_count()) - 1):
                self.memory_manager.set_bool(  # each will switch to 1 when finished
                    f"{WORKER}_{i}", False
                )

        with self.locks.status_lock:
            self.memory_manager.set_str(RUN_ID, self.run_id)
        # with self.counters_lock:  # TODO: unclear if is needed
        #     self.memory_manager.set_int(DISTRIBUTED, 0)

    def _set_distributor_wasm_port(self, port) -> None:
        """"""

        self.queues.distributor_queue.set_destination(port)

    def reset(
        self,
        board: GameBoardT,
        new_tree_cache: Optional[SearchTreeCache] = None,
        run_iteration: int = 0,
    ) -> None:
        """"""

        self._reset_counters()

        self.board = board

        if new_tree_cache is not None:
            self._reuse_node_cache(new_tree_cache)

        else:
            self._set_new_root()

        self.run_id = self._get_run_id(run_iteration)
        with self.locks.status_lock:
            self.memory_manager.set_str(RUN_ID, self.run_id)

        self.traverser = Traverser(self.node_cache)

    def _get_run_id(self, run_iteration: int = 0) -> str:
        """"""

        # TODO: random part is added to decrease chance of collision, but another solution is needed to *never* collide
        # zfill 3 to fill constant space in memory
        return f"{Node, self.node_cache.root.move}.{run_iteration}.{str(randint(0, 999)).zfill(3)}"

    def _reuse_node_cache(self, node_cache: SearchTreeCache) -> None:
        """"""

        self.node_cache = node_cache

        self.node_cache.root.parent = None
        self.node_cache.nodes_dict = node_cache.nodes_dict
        self.node_cache.transposition_dict = node_cache.transposition_dict

    def _create_root(self) -> Node:
        """"""

        return Node(
            parent=None,
            move=self.board.move_stack[-1].uci() if self.board.move_stack else ROOT_NODE_NAME,
            score=-INF/2 if self.board.turn else INF/2,
            forcing_level=0,
            color=self.board.turn,
            board=self.board.serialize_position(),
        )

    def _set_new_root(self) -> None:
        """"""

        serialized_board = self.board.serialize_position()
        self.node_cache.root = self._create_root()

        self.node_cache.nodes_dict = {ROOT_NODE_NAME: self.node_cache.root}
        self.node_cache.transposition_dict = self.node_cache.transposition_dict and {
            serialized_board: self.node_cache.root
        }

    def run(self) -> None:
        """"""

        if self.flags.should_profile:
            from pyinstrument import Profiler  # pylint: disable=import-outside-toplevel

            profiler = Profiler()
            profiler.start()

        self._return = self.search()

        if self.flags.should_profile:
            profiler.stop()
            profiler.print(show_all=True)

    def finish(self) -> None:
        """"""

        self.flags.finished = True
        self.flags.started = False

    async def search(self) -> str:
        """"""

        limit = SEARCH_LIMIT
        if limit == 0:
            self.traverser.should_autodistribute = False
            self.limit = 0
        else:
            if self.board.has_move_limit:
                limit = min(limit, self.board.get_move_limit())

            self.limit = 2**limit  # 14 -> 16384, 15 -> 32768

        # must set status started before putting the element on queue or else will be discarded
        with self.locks.status_lock:
            self.memory_manager.set_int(STATUS, Status.STARTED)

        if not self.node_cache.root.children:  # or self.node_cache.root.only_forcing:
            self.node_cache.root.being_processed = True
            self.queues.distributor_queue.put(
                DistributorItem(
                    self.run_id,
                    ROOT_NODE_NAME,
                    -1 if self.node_cache.root.only_forcing else 0,
                    self.node_cache.root.score,
                    self.board.serialize_position(),
                )
            )
            self.flags.started = True

        t_0: float = self.counters.time
        try:
            # TODO: refactor to use concurrency?
            while not (
                self.flags.finished
                and self.counters.evaluated >= self.counters.distributed
            ):  # TODO: should check for equality maybe, but this is safer
                if await self.main_loop(self.node_cache.root):
                    break
                await asyncio.sleep(0)

        finally:
            self._signal_run_finished()
            await self._wait_all_workers()

        return self.finish_up(time() - t_0)

    async def main_loop(self, root: Node) -> bool:
        """
        :returns: if should break the loop
        """

        if self._has_winning_move():
            # from previous analysis is already known the winning move, let's play it
            self.finish()
            return True

        self._update_counters()

        if self._monitor(root):
            # on a signal to stop the thread
            return True

        if self.counters.distributed == 0:
            if root.children and not self.flags.started:
                t = self._select_from_tree(min((4, len(root.children))))
                print("selecting")
                if not t:
                    # TODO: leafs sometimes are left `being_processed=True`, fix it
                    print("resetting tree")
                    self.print_tree(0, 3)
                    root.propagate_being_processed_down()
                else:
                    self.flags.started = True

                return False

            # waiting for first eval, check control queue as may not have needed distributing
            try:
                return self._handle_control_queue(timeout=SLEEP)
            except:
                print(
                    "raised when: ",
                    self.flags.started,
                    self.run_id,
                    root.children,
                )
                raise

        i = 0
        while self.balance_load(i):
            i += 1

        if not self.flags.finished and self._is_enough():
            self.finish()

        return False

    def _update_counters(self) -> None:
        """
        Update counters values published by other processes.

        :returns: if system is ready to proceed
        """

        with self.locks.counters_lock:
            distributed = self.memory_manager.get_int(DISTRIBUTED)
            if distributed != self.counters.last_external_distributed:
                self.counters.last_external_distributed = distributed
                self.counters.distributed = distributed

    def _monitor(self, root: Node) -> bool:
        """
        Monitor search progress and log events.

        :returns: if the process hang and should be finished
        """

        t = time()

        if (
            t > self.counters.time + LOG_INTERVAL and self.limit != 0
        ):
            # if self.printing not in [Print.NOTHING, Print.MOVE, Print.LOGS]:
            #     os.system("clear")

            progress = (
                round(
                    min(
                        self.counters.evaluated / self.counters.distributed,
                        self.counters.evaluated / self.limit,
                    )
                    * 100
                )
                if self.counters.distributed > 0
                else 0
            )

            if (
                progress > 0
                and self.counters.evaluated == self.counters.last_evaluated
                and self.counters.distributed == self.counters.last_distributed
            ):
                # TODO: use signal for this?
                if not self.flags.finished:
                    if self.counters.evaluated == self.counters.distributed:
                        # TODO: a failsafe for now, but find out why this happens for Hex
                        print(f"finished with only {self.counters.evaluated} evaluated")
                        return True
                    print(
                        f"distributed: {self.counters.distributed}, evaluated: {self.counters.evaluated}, "
                        f"selected: {self.counters.selected}, started: {self.flags.started}, "
                        f"finished: {self.flags.finished}"
                    )
                    # self.print_tree(0, 4)
                    # self.memory_manager.set_int(DEBUG, 1)
                    with self.locks.status_lock:
                        self.memory_manager.set_int(STATUS, Status.FINISHED, new=False)

                    # self.debug = True
                    raise SearchFailed("nodes not delivered on queues")

            if PRINTING not in [Print.MOVE, Print.NOTHING]:
                print(
                    f"distributed: {self.counters.distributed}, evaluated: {self.counters.evaluated}, "
                    f"gap: {self.counters.distributed - self.counters.evaluated}, "
                    f"selected: {self.counters.selected}, explored: {self.counters.explored}, progress: {progress}%"
                )

            if PRINTING in [Print.CANDIDATES, Print.TREE]:
                sorted_children: List[Node] = sorted(
                    [child for child in root.children if not child.only_forcing],
                    key=lambda node: node.score,
                    reverse=root.color,
                )

                for child in sorted_children[:PRINT_CANDIDATES]:
                    print(
                        child.move, child.leaf_level, child.score, child.being_processed
                    )

            self.counters.time = t
            self.counters.last_evaluated = self.counters.evaluated
            self.counters.last_distributed = self.counters.distributed

        return self._stop_event.is_set()

    def _has_winning_move(self) -> bool:
        """"""

        return np_abs(self.node_cache.root.score) == INF

    def balance_load(self, i: int) -> bool:
        """
        Prevent queues from starving. Loops to clear the selector queue ASAP.

        Choosing between actions:
            - pick up from `selector_queue` and handle evaluated nodes
            - pick from `control_queue` and handle special nodes (currently nodes with 0/1 children)
            - publish to `distributor_queue` to order evaluation of best nodes

        :returns: if should repeat
        """

        max_gap = 2000  # TODO: should dynamically change based on evaluation speed
        gap = self.counters.distributed - self.counters.evaluated
        """
        Goal is to keep this value on a relatively constant level appropriate to processing speed. 
        Most importantly it should never reach 0 before the end of processing.
        """

        if i > 10000:  # found in practice that levels below happen normally (not stuck)
            print(
                "balance load: ",
                gap,
                self.counters.distributed,
                self.counters.evaluated,
            )

        # TODO: find smart conditions instead of that mess - purpose is to identify if should distribute more to eval
        if not self.flags.finished and self.limit > 0 and gap < max_gap:
            # feed the eval queues
            # return self._select_from_tree(  # TODO: the param should depend on both gap and speed
            #     (24000 - gap) // 4000 + 1,  # effectively between 1 and 6
            # )
            if not self._select_from_tree(  # TODO: the param should depend on both gap and speed
                (24000 - gap) // 4000 + 1,  # effectively between 1 and 6
            ):
                self._handle_control_queue()
                self._handle_selector_queue()
                return False  # breaking the loop

            return True

        # TODO: should it decide between two queues based on something?
        self._handle_control_queue()
        queue_emptied = not self._handle_selector_queue()

        return not (
            self.flags.finished or queue_emptied
        )  # breaking the loop if finished or queue empty

    def _is_enough(self) -> bool:
        """"""

        return (
            self.counters.distributed > self.limit
            or np_abs(self.node_cache.root.score) + 1 > INF
        )  # is checkmate

    def _signal_run_finished(self) -> None:
        """
        Send msg to distributor so that it resets counters.
        """

        with self.locks.status_lock:
            self.memory_manager.set_int(STATUS, Status.FINISHED, new=False)
        # self.queues.distributor_queue.put(DistributorItem("finished", "", float32(0), 0))

    def _handle_selector_queue(self) -> bool:
        """
        :returns: if items found on queue
        """

        candidates: List[SelectorItem] = self._exclude_by_run_id(
            self.queues.selector_queue.get_many(QUEUE_THROTTLE, SLEEP)
        )
        if not candidates:
            # print(f"no items")
            return False

        self.counters.evaluated += len(candidates)

        self.handle_candidates(self.queues.distributor_queue, candidates)

        return True

    def _exclude_by_run_id(self, candidates: List[SelectorItem]) -> List[SelectorItem]:
        """"""

        def is_right_id(item):
            if item.run_id == self.run_id:
                return True

            return False

        return [item for item in candidates if is_right_id(item)]

    def _handle_control_queue(self, timeout: float = 0.0) -> bool:
        """
        :returns: True if should finish the search, False otherwise
        """

        control_items: List[ControlItem] = self.queues.control_queue.get_many(
            1000, timeout=timeout
        )
        if not control_items:
            return False

        for item in control_items:
            if item.control_value == "error":  # TODO: switch to read status from memory
                raise SearchFailed("Distributor error")
            if item.run_id != self.run_id:
                # value from previous cycle
                continue

            # only 1 root child, therefore nothing to analyse (finishing immediately)
            if item.control_value == ROOT_NODE_NAME:
                # self.evaluated += 1
                move = list(self.board.legal_moves)[0]

                self.board.push(move)
                self.traverser.create_node(
                    parent=self.node_cache.root,
                    move=move.uci(),
                    score=float32(0),
                    forcing_level=0,
                    color=not self.node_cache.root.color,
                    board=self.board.serialize_position(),
                )
                self.board.pop()

                self.finish()
                print("only 1 root child")
                return True

            # 0 children, so nothing sent to evaluation
            node = self.traverser.get_node(item.control_value)

            # TODO: if no children then control if the score is valid for checkmate or stalemate
            # propagate here at the end of capture-fest
            node.being_processed = False
            node.propagate_being_processed_up()
            node.parent.inherit_score(node.score, None)

        return False

    async def _wait_all_workers(self) -> None:
        """"""

        # print("wait all workers")
        for i in range((PROCESS_COUNT or cpu_count()) - 1):
            attempts = 0
            while (
                attempts < 1000
            ):  # TODO: why processes take so long to set new status?
                attempts += 1
                with self.locks.finish_lock:
                    worker_status = self.memory_manager.get_bool(f"{WORKER}_{i}")

                if not worker_status:
                    # sleep(SLEEP)
                    await asyncio.sleep(SLEEP)
                else:
                    break
            else:
                print("exceeded waiting time")

        # print("workers done")

    def handle_candidates(
        self, distributor_queue: QM[DistributorItem], candidates: List[SelectorItem]
    ) -> None:
        """"""

        try:
            nodes_to_distribute: List[Node] = (
                self.traverser.create_nodes_and_autodistribute(candidates)
            )
        except SearchFailed:
            self.print_tree(0, 2)
            raise

        if nodes_to_distribute:
            # TODO: revise this magic that prevented extremely deep capture fest
            # if self.finished and self.node_cache.root.children:
            #     top_node = (
            #         max(self.node_cache.root.children, key=lambda node: node.score)
            #         if self.node_cache.root.children
            #         else min(self.node_cache.root.children, key=lambda node: node.score)
            #     )
            #     if not top_node.being_processed or self.distributed > 3 * self.limit:
            #         return

            self.queue_for_distribution(
                distributor_queue, nodes_to_distribute, forcing_moves_only=True
            )

    def _select_from_tree(self, iterations: int = 1) -> bool:
        """
        Select next nodes to explore and queue for distribution.

        :returns: if something was distributed
        """

        top_leafs = self.traverser.get_leafs_to_look_at(iterations)
        if not top_leafs:
            return False

        self.queue_for_distribution(
            self.queues.distributor_queue, top_leafs, forcing_moves_only=False
        )
        return True

    def queue_for_distribution(
        self,
        distributor_queue: QM[DistributorItem],
        nodes: List[Node],
        *,
        forcing_moves_only: bool,
    ) -> None:
        """"""

        to_queue: List[DistributorItem] = []
        for node in nodes:
            forcing_level: int
            if forcing_moves_only:
                forcing_level = node.forcing_level
                node.only_forcing = True
            elif node.only_forcing:  # forcing moves have already been distributed
                forcing_level = (
                    -1
                )  # this will indicate that only non-forcing moves are generated
                node.only_forcing = False
            else:
                forcing_level = 0

            # print("q", node.name)
            to_queue.append(
                DistributorItem(
                    self.run_id,
                    node.name,
                    forcing_level,
                    node.score,
                    node.board,
                )
            )

        n_nodes: int = len(to_queue)
        if not forcing_moves_only:
            self.counters.selected += n_nodes
        self.counters.explored += n_nodes
        self.counters.distributed += (
            self.board.size_square
            * n_nodes  # roughly, in fact children of those n nodes are distributed
        )

        distributor_queue.put_many(to_queue)

    def finish_up(self, total_time: float) -> str:
        """"""

        if PRINTING == Print.TREE:
            min_depth, max_depth, path = TREE_PARAMS.split(",")
            self.print_tree(int(min_depth), int(max_depth), path)

        best_score, best_move = self.get_best_move()

        if PRINTING not in [Print.NOTHING, Print.MOVE]:
            if PRINTING != Print.MOVE:
                print(
                    f"distributed: {self.counters.distributed}, evaluated: {self.counters.evaluated}, "
                    f"selected: {self.counters.selected}, explored: {self.counters.explored}"
                )

                print(
                    f"time: {total_time}, nodes/s: {round(self.counters.evaluated / total_time)}"
                )

            print("chosen move -->", round(best_score, 3), best_move)
        elif PRINTING == Print.MOVE:
            print(best_move)

        self._reset_counters()

        # print(sorted([(node.name, score) for node, score in self.traverser.selections.items()], key=lambda x: x[1],
        #              reverse=True))

        return best_move

    def get_best_move(self) -> Tuple[float32, str]:
        """
        Get the first move with the highest score with respect to color.
        """

        if not self.node_cache.root.children:
            print(
                f"distributed: {self.counters.distributed}, evaluated: {self.counters.evaluated}, "
                f"selected: {self.counters.selected}, finished: {self.flags.finished}"
            )
            self.print_tree(0, 2)
            raise SearchFailed("finished without analysis")

        sorted_children: List[Node] = sorted(
            self.node_cache.root.children,
            key=lambda node: node.score,
            reverse=self.node_cache.root.color,
        )

        if PRINTING == Print.CANDIDATES:
            print("***")
            os.system("clear")
            for child in sorted_children[:]:
                print(child.move, child.leaf_level, child.score)

        depth = max(  # pylint: disable=consider-using-generator
            [child.leaf_level for child in sorted_children[:3]]
        )

        for child in sorted_children:
            if child.leaf_level >= 2 / 3 * depth:
                best = child
                break
        return best.score, best.move

    def print_tree(
        self, depth_from: int = 0, depth_to: int = 100, path_constraint: str = ""
    ) -> None:
        """"""

        print(
            PrunedTreeRenderer(
                self.node_cache.root,
                depth=depth_from,
                maxlevel=depth_to,
                path=path_constraint,
            )
        )
