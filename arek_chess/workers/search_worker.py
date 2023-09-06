# -*- coding: utf-8 -*-
import os
from multiprocessing import Lock, cpu_count
from random import randint
from time import perf_counter, sleep, time
from typing import ClassVar, Dict, Generic, List, Optional, Tuple, TypeVar

from numpy import abs, float32

from arek_chess.board import GameBoardBase
from arek_chess.common.constants import (
    DEBUG,
    DISTRIBUTED,
    INF,
    LOG_INTERVAL,
    PRINT_CANDIDATES,
    Print,
    ROOT_NODE_NAME,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    WORKER,
)
from arek_chess.common.custom_threads import ReturningThread
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory.manager import MemoryManager
from arek_chess.common.profiler_mixin import ProfilerMixin
from arek_chess.common.queue.items.control_item import ControlItem
from arek_chess.common.queue.items.distributor_item import DistributorItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager as QM
from arek_chess.game_tree.node import Node
from arek_chess.game_tree.renderer import PrunedTreeRenderer
from arek_chess.game_tree.traverser import Traverser

TGameBoard = TypeVar("TGameBoard", bound=GameBoardBase)


class SearchWorker(ReturningThread, ProfilerMixin, Generic[TGameBoard]):
    """
    Handles the in-memory game tree and controls all tree expansion logic and progress logging.
    """

    board: TGameBoard
    root: Optional[Node]
    run_id: str

    status_lock: Lock
    counters_lock: Lock

    tree: Dict[str, Node]
    distributor_queue: QM[DistributorItem]
    selector_queue: QM[SelectorItem]
    control_queue: QM[ControlItem]
    queue_throttle: int
    printing: Print
    tree_params: str
    limit: int
    should_profile: bool

    distributed: int
    evaluated: int
    selected: int
    explored: int

    debug_log: ClassVar[List] = []

    def __init__(
        self,
        status_lock: Lock,
        finish_lock: Lock,
        counters_lock: Lock,
        distributor_queue: QM[DistributorItem],
        selector_queue: QM[SelectorItem],
        control_queue: QM[ControlItem],
        queue_throttle: int,
        printing: Print,
        tree_params: str,
    ):
        super().__init__()

        self.distributor_queue = distributor_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = queue_throttle

        self.printing = printing
        self.tree_params = tree_params  # TODO: create a constant class like Print

        self.root = None
        # self.nodes_dict: Dict[str, Node] = {}
        self.nodes_dict: Dict[str, Node] = {}  # WeakValueDictionary({})
        self.transposition_dict: Optional[
            Dict[bytes, Node]
        ] = None  # WeakValueDictionary({})
        self.memory_manager = MemoryManager()
        self.status_lock = status_lock
        self.finish_lock = finish_lock
        self.counters_lock = counters_lock

        self._reset_counters()

        self.started: bool = False
        self.finished: bool = False
        self.should_profile: bool = False

        self.debug = False

        with self.status_lock:
            self.memory_manager.set_int(STATUS, Status.CLOSED)

    def _reset_counters(self) -> None:
        """"""

        self.distributed = 0
        self.evaluated = 0
        self.selected = 0
        self.explored = 0

        self.started = False
        self.finished = False

        with self.status_lock:
            self.memory_manager.set_int(STATUS, Status.CLOSED)
            self.memory_manager.set_int(DEBUG, 0)

        with self.finish_lock:
            for i in range(cpu_count() - 1):
                self.memory_manager.set_int(
                    f"{WORKER}_{i}", 0
                )  # each will switch to 1 when finished

        # with self.counters_lock:  # TODO: unclear if is needed
        #     self.memory_manager.set_int(DISTRIBUTED, 0)

    def reset(
        self,
        board: TGameBoard,
        new_root: Optional[Node] = None,
        nodes_dict: Dict[str, Node] = None,
        run_iteration: int = 0,
        should_use_transposition: bool = True,
    ) -> None:
        """"""

        self._reset_counters()

        self.board = board

        if new_root and nodes_dict is not None:
            self._reuse_root(new_root, nodes_dict)

        else:
            self._set_new_root(should_use_transposition)

        self.run_id = (
            f"{self.root.move}.{run_iteration}"
            if self.root.move != "1"
            else f"{randint(2, 100)}.{run_iteration}"  # in case of running root multiple times have different run_id
        )
        # print("searching with run_id: ", self.run_id)
        with self.status_lock:
            self.memory_manager.set_str(RUN_ID, self.run_id)

        self.traverser: Traverser = Traverser(
            self.root, self.nodes_dict, self.transposition_dict
        )

    def _reuse_root(
        self,
        new_root: Node,
        nodes_dict: Dict,
    ) -> None:
        """"""

        # TODO: must be a bug in setting being_processed back to False for nodes after processing
        # new_root.propagate_being_processed_down()  # TODO: can this run in infinite loop?...

        self.root = new_root

        self.root.parent = None  # gc.collect needed ?
        self.nodes_dict = nodes_dict

    def _set_new_root(self, should_use_transposition: bool = True) -> None:
        """"""

        score = -INF if self.board.turn else INF
        move = (
            self.board.move_stack[-1].uci() if self.board.move_stack else ROOT_NODE_NAME
        )

        serialized_board = self.board.serialize_position()
        self.root = Node(
            parent=None,
            move=move,
            score=score / 2,  # divided by 2 to differentiate from checkmate
            forcing_level=0,
            color=self.board.turn,
            board=serialized_board,
        )

        self.nodes_dict = {ROOT_NODE_NAME: self.root}
        self.transposition_dict = (
            {serialized_board: self.root} if should_use_transposition else None
        )

        if should_use_transposition:
            print("****** TRANSPOSITIONS ON ********")

    def run(self) -> None:
        """"""

        if self.should_profile:
            from pyinstrument import Profiler

            self.profiler = Profiler()
            self.profiler.start()

        self._return = self.search()

        if self.should_profile:
            self.profiler.stop()
            self.profiler.print(show_all=True)

    def finish(self) -> None:
        """"""

        self.finished = True
        self.started = False

    def search(self, limit: int = 14) -> str:
        """"""

        self.limit: int = 2 ** limit  # 14 -> 16384, 15 -> 32768
        if self.board.has_move_limit:
            uppper_limit = self.board.get_move_limit()
            if uppper_limit < self.limit:
                self.limit = uppper_limit  # TODO: maybe doesn't make sense because at limit someone wins anyway

        if limit == 0:
            self.traverser.should_autodistribute = False

        # must set status started before putting the element on queue or else will be discarded
        with self.status_lock:
            self.memory_manager.set_int(STATUS, Status.STARTED)

        if not self.root.children:  # or self.root.only_forcing:
            self.root.being_processed = True
            self.distributor_queue.put(
                DistributorItem(
                    self.run_id,
                    ROOT_NODE_NAME,
                    -1 if self.root.only_forcing else 0,
                    self.root.score,
                    self.board.serialize_position(),
                )
            )
            self.started = True

        self.t_0: float = time()
        self.t_tmp: float = self.t_0
        self.last_evaluated: int = 0
        self.last_distributed: int = 0

        try:
            # TODO: refactor to use concurrency?
            while not (
                self.finished and self.evaluated >= self.distributed
            ):  # TODO: should check for equality maybe, but this is safer
                if self.main_loop():
                    break

        finally:
            self._signal_run_finished()
            self._wait_all_workers()

        # with open(self.run_id, "w") as f:
        #     for l in Node.debug_log:
        #         f.write(", ".join(l))
        #
        # Node.debug_log = []

        return self.finish_up(time() - self.t_0)

    def main_loop(self) -> bool:
        """
        :returns: if should break the loop
        """

        if self._has_winning_move():
            # from previous analysis is already known the winning move, let's play it
            self.finish()
            return True

        self._update_counters()

        if self._monitor():
            # on a signal to stop the thread
            return True

        if self.distributed == 0:
            if self.root.children and not self.started:
                t = self._select_from_tree(min((4, len(self.root.children))))
                print("selecting")
                if not t:
                    # TODO: leafs sometimes are left `being_processed=True`, fix it
                    print("resetting tree")
                    self.print_tree(0, 3)
                    self.root.propagate_being_processed_down()
                else:
                    self.started = True

                return False

            # waiting for first eval, check control queue as may not have needed distributing
            return self._handle_control_queue(timeout=SLEEP)

        i = 0
        while self.balance_load(i):
            i += 1
            ...

        if not self.finished and self._is_enough():
            self.finish()

        return False

    def _update_counters(self) -> None:
        """
        Update counters values published by other processes.

        :returns: if system is ready to proceed
        """

        with self.counters_lock:
            self.distributed = self.memory_manager.get_int(DISTRIBUTED)

    def _monitor(self) -> bool:
        """
        Monitor search progress and log events.

        :returns: if the process hang and should be finished
        """

        t = time()

        if t > self.t_tmp + LOG_INTERVAL:
            if self.printing not in [Print.NOTHING, Print.MOVE, Print.LOGS]:
                os.system("clear")

            progress = (
                round(
                    min(
                        self.evaluated / self.distributed,
                        self.evaluated / self.limit,
                    )
                    * 100
                )
                if self.distributed > 0
                else 0
            )

            if (
                progress > 0
                and self.evaluated == self.last_evaluated
                and self.distributed == self.last_distributed
            ):
                # TODO: use signal for this?
                if not self.finished:
                    if self.evaluated == self.distributed:
                        # TODO: a failsafe for now, but find out why this happens for Hex
                        print(f"finished with only {self.evaluated} evaluated")
                        return True
                    print(
                        f"distributed: {self.distributed}, evaluated: {self.evaluated}, "
                        f"selected: {self.selected}, started: {self.started}, finished: {self.finished}"
                    )
                    self.print_tree(0, 4)
                    self.memory_manager.set_int(DEBUG, 1)
                    with self.status_lock:
                        self.memory_manager.set_int(STATUS, Status.FINISHED, new=False)

                    # self.debug = True
                    raise SearchFailed("nodes not delivered on queues")

            if self.printing not in [Print.MOVE, Print.NOTHING]:
                print(
                    f"distributed: {self.distributed}, evaluated: {self.evaluated}, "
                    f"gap: {self.distributed - self.evaluated}, "
                    f"selected: {self.selected}, explored: {self.explored}, progress: {progress}%"
                )

            if self.printing in [Print.CANDIDATES, Print.TREE]:
                sorted_children: List[Node] = sorted(
                    [child for child in self.root.children if not child.only_forcing],
                    key=lambda node: node.score,
                    reverse=self.root.color,
                )

                for child in sorted_children[:PRINT_CANDIDATES]:
                    print(
                        child.move, child.leaf_level, child.score, child.being_processed
                    )

            self.t_tmp = t
            self.last_evaluated = self.evaluated
            self.last_distributed = self.distributed

        return self._stop_event.is_set()

    def _has_winning_move(self) -> bool:
        """"""

        return abs(self.root.score) == INF

    def balance_load(self, i: int) -> bool:
        """
        Prevent queues from starving.

        Choosing between actions:
            - pick up from `selector_queue` and handle evaluated nodes
            - pick from `control_queue` and handle special nodes (currently nodes with 0/1 children)
            - publish to `distributor_queue` to order evaluation of best nodes

        :returns: if should repeat
        """

        gap = self.distributed - self.evaluated
        """
        Goal is to keep this value on a relatively constant level appropriate to processing speed. 
        Most importantly it should never reach 0 before the end of processing.
        """

        gap_ratio = gap / self.distributed  # already checked to not be 0

        if i > 1000:  # found in practice that levels below happen normally (not stuck)
            print("balance load: ", gap, gap_ratio, self.distributed, self.queue_throttle)

        # TODO: find smart conditions instead of that mess
        if not self.finished and (
            (gap_ratio < 0.1 and gap < 4000)  # always pump when little gap
            or (
                gap < 24000  # above this number is impractical to add more
                and not gap_ratio
                > 0.5  # evaluated less than half, let's not pump it more
                and not (
                    self.distributed
                    < self.limit // 2  # early part of the analysis doesn't need to rush
                    and gap > 10000
                )
            )
        ):
            if not self._select_from_tree(  # TODO: the param should depend on both gap and speed
                (24000 - gap) // 4000 + 1,  # effectively between 1 and 6
            ):
                self._handle_control_queue()
                self._handle_selector_queue()

            return False

        # TODO: should it decide between two queues based on something?
        self._handle_control_queue()
        self._handle_selector_queue()

        return not self.finished

    def _is_enough(self) -> bool:
        """"""

        return (
            self.distributed > self.limit or abs(self.root.score) + 1 > INF
        )  # is checkmate

    def _signal_run_finished(self) -> None:
        """
        Send msg to distributor so that it resets counters.
        """

        with self.status_lock:
            self.memory_manager.set_int(STATUS, Status.FINISHED, new=False)
        # self.distributor_queue.put(DistributorItem("finished", "", float32(0), 0))

    def _handle_selector_queue(self) -> None:
        """"""

        candidates: List[SelectorItem] = self._exclude_by_run_id(
            self.selector_queue.get_many(self.queue_throttle, SLEEP)
        )
        if not candidates:
            # print(f"no items")
            return

        self.evaluated += len(candidates)

        self.handle_candidates(self.distributor_queue, candidates)

    def _exclude_by_run_id(self, candidates: List[SelectorItem]) -> List[SelectorItem]:
        """"""

        def is_right_id(item):
            if item.run_id == self.run_id:
                return True
            else:
                # print(item.run_id, item.node_name, item.move_str)
                return False

        return [item for item in candidates if is_right_id(item)]

    def _handle_control_queue(self, timeout: float = 0.0) -> bool:
        """
        :returns: True if should finish the search, False otherwise
        """

        control_items: List[ControlItem] = self.control_queue.get_many(1000, timeout=timeout)
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
                    parent=self.root,
                    move=move.uci(),
                    score=float32(0),
                    forcing_level=0,
                    color=not self.root.color,
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
            node.parent.propagate_score(node.score, None)

        return False

    def _wait_all_workers(self) -> None:
        """"""

        # print("wait all workers")
        for i in range(cpu_count() - 1):
            attempts = 0
            while attempts < 1000:  # TODO: why processes take so long to set new status?
                attempts += 1
                with self.finish_lock:
                    worker_status = self.memory_manager.get_int(f"{WORKER}_{i}")

                if not worker_status:
                    sleep(SLEEP)
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
            nodes_to_distribute: List[
                Node
            ] = self.traverser.create_nodes_and_autodistribute(candidates)
        except SearchFailed:
            self.print_tree(0, 2)
            raise

        if nodes_to_distribute:
            # TODO: revise this magic that prevented extremely deep capture fest
            # if self.finished and self.root.children:
            #     top_node = (
            #         max(self.root.children, key=lambda node: node.score)
            #         if self.root.children
            #         else min(self.root.children, key=lambda node: node.score)
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
            self.distributor_queue, top_leafs, forcing_moves_only=False
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

        # distributor_queue.put_many(
        #     [
        #         DistributorItem(
        #             self.run_id,
        #             node.name,
        #             node.move,
        #             node.captured if recaptures else -1 if node.only_forcing else 0,
        #             node.score,
        #             node.board
        #         )
        #         for node in nodes
        #     ]
        # )

        to_queue: List[DistributorItem] = []
        for node in nodes:
            forcing_level: int
            if forcing_moves_only:
                forcing_level = node.forcing_level
                node.only_forcing = True
            elif node.only_forcing:  # forcing moves have already been distributed
                forcing_level = -1  # this will indicate that only non-forcing moves are generated
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
            self.selected += n_nodes
        self.explored += n_nodes
        self.distributed += n_nodes

        distributor_queue.put_many(to_queue)

    def finish_up(self, total_time: float) -> str:
        """"""

        if self.printing == Print.TREE:
            min_depth, max_depth, path = self.tree_params.split(",")
            self.print_tree(int(min_depth), int(max_depth), path)

        best_score, best_move = self.get_best_move()

        if self.printing not in [Print.NOTHING, Print.MOVE]:
            if self.printing != Print.MOVE:
                print(
                    f"distributed: {self.distributed}, evaluated: {self.evaluated}, "
                    f"selected: {self.selected}, explored: {self.explored}"
                    # f"selected: {','.join([node[0].name for node in self.selected_nodes])}, explored: {self.explored}"
                )

                print(
                    f"time: {total_time}, nodes/s: {round(self.evaluated / total_time)}"
                )

                # print(sorted([(node.name, score) for node, score in self.traverser.selections.items()], key=lambda x: x[1], reverse=True))

            print("chosen move -->", round(best_score, 3), best_move)
        elif self.printing == Print.MOVE:
            print(best_move)

        self.distributed = 0
        self.evaluated = 0
        self.selected = 0
        self.explored = 0

        # print(sorted([(node.name, score) for node, score in self.traverser.selections.items()], key=lambda x: x[1],
        #              reverse=True))

        return best_move

    def get_best_move(self) -> Tuple[float32, str]:
        """
        Get the first move with the highest score with respect to color.
        """

        if not self.root.children:
            print(
                f"distributed: {self.distributed}, evaluated: {self.evaluated}, selected: {self.selected}, finished: {self.finished}"
            )
            self.print_tree(0, 2)
            raise SearchFailed("finished without analysis")

        sorted_children: List[Node] = sorted(
            self.root.children, key=lambda node: node.score, reverse=self.root.color
        )

        if self.printing == Print.CANDIDATES:
            print("***")
            os.system("clear")
            for child in sorted_children[:]:
                print(child.move, child.leaf_level, child.score)

        depth = max([child.leaf_level for child in sorted_children[:3]])

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
                self.root, depth=depth_from, maxlevel=depth_to, path=path_constraint
            )
        )
