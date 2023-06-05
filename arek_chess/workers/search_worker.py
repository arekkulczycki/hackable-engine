# -*- coding: utf-8 -*-
import gc
import os
from asyncio import sleep as asyncio_sleep
from time import time, sleep
from typing import Optional, Dict, List, Tuple
from weakref import WeakValueDictionary

from numpy import abs, float32

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    Print,
    INF,
    ROOT_NODE_NAME,
    LOG_INTERVAL,
    FINISHED,
    STARTED,
    STATUS,
    DISTRIBUTED,
    SLEEP,
    CLOSED,
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

N_CANDIDATES = 10


class SearchWorker(ReturningThread, ProfilerMixin):
    """
    Handles the in-memory game tree and controls all tree expansion logic and progress logging.
    """

    board: Board
    root: Node
    run_id: str

    tree: Dict[str, Node]
    distributor_queue: QM
    selector_queue: QM
    control_queue: QM
    queue_throttle: int
    printing: Print
    tree_params: str
    limit: int
    should_profile: bool

    def __init__(
        self,
        distributor_queue: QM,
        selector_queue: QM,
        control_queue: QM,
        queue_throttle: int,
        printing: Print,
        tree_params: str,
        search_limit: Optional[int],
    ):
        super().__init__()

        self.distributor_queue = distributor_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = queue_throttle

        self.printing = printing
        self.tree_params = tree_params  # TODO: create a constant class like Print

        self.root: Optional[Node] = None
        # self.nodes_dict: Dict[str, Node] = {}
        self.nodes_dict: WeakValueDictionary = WeakValueDictionary({})
        self.memory_manager = MemoryManager()

        self._reset_counters()

        self.limit: int = 2 ** (search_limit or 14)  # 14 -> 16384, 15 -> 32768

        self.should_profile: bool = False

    def _reset_counters(self) -> None:
        """"""

        self.distributed: int = 0
        self.evaluated: int = 0
        self.selected: int = 0
        self.explored: int = 0

        self.finished: bool = False

        self.memory_manager.set_int(STATUS, CLOSED)

    def restart(
        self,
        board: Board,
        new_root: Optional[Node] = None,
        nodes_dict: WeakValueDictionary = None,
        run_iteration: int = 0,
    ) -> None:
        """"""

        self._reset_counters()

        self.board = board

        if new_root and nodes_dict:
            self._reuse_root(new_root, nodes_dict)

        else:
            self._set_new_root()

        self.run_id = f"{self.root.move}.{run_iteration}"

        if not self.root.children:
            self.distributor_queue.put(
                DistributorItem(
                    self.run_id,
                    ROOT_NODE_NAME,
                    self.root.move,
                    0,
                    self.root.score,
                    self.board.serialize_position(),
                )
            )

        self.traverser: Traverser = Traverser(self.root, self.nodes_dict)

    def _set_new_root(self) -> None:
        """"""

        score = -INF if self.board.turn else INF
        move = (
            self.board.move_stack[-1].uci() if self.board.move_stack else ROOT_NODE_NAME
        )

        self.root = Node(
            parent=None,
            move=move,
            score=score / 2,  # divided by 2 to differentiate from checkmate
            captured=0,
            color=self.board.turn,
            being_processed=True,
            only_captures=False,
            board=self.board.serialize_position()
        )

        self.nodes_dict = {ROOT_NODE_NAME: self.root}

    def _reuse_root(self, new_root: Node, nodes_dict: WeakValueDictionary) -> None:
        """"""

        # del self.root
        # gc.collect()

        self.root = new_root
        self.root.parent = None
        self.nodes_dict = nodes_dict
        # print(f"inheriting {len(self.nodes_dict.keys())} nodes")

    def run(self) -> None:
        """"""

        if self.should_profile:
            from pyinstrument import Profiler

            self.profiler = Profiler()
            self.profiler.start()

        self._return = self._search()

        if self.should_profile:
            self.profiler.stop()
            self.profiler.print(show_all=True)

    def _search(self) -> str:
        """"""

        self.memory_manager.set_int(STATUS, STARTED)
        self.memory_manager.set_int(DISTRIBUTED, 0)

        self.t_0: float = time()
        self.t_tmp: float = self.t_0
        self.last_evaluated: int = 0

        limit = self.limit
        queue_throttle = self.queue_throttle
        distributor_queue = self.distributor_queue
        selector_queue = self.selector_queue
        control_queue = self.control_queue
        read_control_values = self._read_control_values
        harvest_queues = self.harvest_queues
        select_from_tree = self.select_from_tree
        monitor = self.monitor

        try:
            # TODO: refactor to use concurrency?
            while not (
                self.finished and self.evaluated >= self.distributed
            ):  # TODO: should check for equality but elements from previous runs are consumed too late
                read_control_values()
                harvest_queues(
                    selector_queue, distributor_queue, control_queue, queue_throttle
                )

                if not self.finished:
                    gap = 0
                    ratio = 0
                    if self.distributed != 0:
                        gap = self.distributed - self.evaluated
                        ratio = gap / self.distributed
                    if self.distributed == 0 or (
                        gap < 25000
                        and not (ratio > 0.25 and gap > 10000)
                        and not (
                            ratio > 0.1
                            and gap > 10000
                            and self.distributed
                            < limit / 2  # slow down in the first half of the search
                        )
                    ):
                        select_from_tree(  # TODO: the param should depend on both gap and speed
                            distributor_queue,
                            1
                            if self.distributed < 10000 or gap > 20000
                            else 2
                            if gap > 10000
                            else 4
                            if gap > 4000
                            else 6,
                        )

                if monitor():
                    # breaks on a signal to stop the thread
                    break

                if not self.finished and self._is_enough():
                    self.finished = True
                    self._run_validation_loop(distributor_queue)

        finally:
            self._reset_distributor()
            self._wait_closed_status()

        return self.finish_up(time() - self.t_0)

    def monitor(self) -> bool:
        """
        Monitor search progress and log events.

        :return: if the process hang and should be finished
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

            if self.evaluated == self.last_evaluated:
                if not self.finished or progress < 95:
                    print(
                        f"distributed: {self.distributed}, evaluated: {self.evaluated}, selected: {self.selected}"
                    )
                    # print(PrunedTreeRenderer(self.root, depth=0, maxlevel=20))
                    raise SearchFailed("nodes not delivered on queues")
                else:
                    # TODO: should be gone when queues handling is fixed and all items are consumed the right time
                    return True

            if self.printing not in [Print.MOVE, Print.NOTHING]:
                print(
                    f"distributed: {self.distributed}, evaluated: {self.evaluated}, "
                    f"gap: {self.distributed - self.evaluated}, "
                    f"selected: {self.selected}, explored: {self.explored}, progress: {progress}%"
                )

            if self.printing in [Print.CANDIDATES, Print.TREE]:
                sorted_children: List[Node] = sorted(
                    self.root.children,
                    key=lambda node: node.score,
                    reverse=self.root.color,
                )

                for child in sorted_children[:N_CANDIDATES]:
                    print(child.move, child.leaf_level, child.score)

            self.t_tmp = t
            self.last_evaluated = self.evaluated

        return self._stop_event.is_set()

    def _is_enough(self) -> bool:
        """"""

        return (
            self.distributed > self.limit or abs(self.root.score) + 1 > INF
        )  # is checkmate

    def _reset_distributor(self) -> None:
        """
        Send msg to distributor so that it resets counters.
        """

        self.memory_manager.set_int(STATUS, FINISHED, new=False)
        # self.distributor_queue.put(DistributorItem("finished", "", float32(0), 0))

    def _run_validation_loop(self, distributor_queue: QM) -> None:
        """"""

        self.select_from_tree(distributor_queue, 2)

    def harvest_queues(
        self,
        selector_queue: QM,
        distributor_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        self._handle_control_queue(control_queue)

        candidates: List[SelectorItem] = self._exclude_by_run_id(
            selector_queue.get_many(queue_throttle, SLEEP)
        )
        if not candidates:
            # print(f"no items")
            return

        self.evaluated += len(candidates)

        self.handle_candidates(distributor_queue, candidates)

    def _exclude_by_run_id(self, candidates: List[SelectorItem]) -> List[SelectorItem]:
        """"""

        return [item for item in candidates if item.run_id == self.run_id]

    def _handle_control_queue(self, control_queue) -> None:
        """"""

        control_items: List[ControlItem] = control_queue.get_many(25, SLEEP)
        for item in control_items:
            if item.control_value == "error":  # TODO: switch to read status from memory
                raise SearchFailed("Distributor error")
            if item.run_id != self.run_id:
                # value from previous cycle
                continue

            # only 1 root child, therefore nothing to analyse (finishing immediately)
            if item.control_value == ROOT_NODE_NAME:
                move = list(self.board.legal_moves)[0]

                self.board.push(move)
                self.traverser.create_node(
                    self.root,
                    move.uci(),
                    float32(0),
                    0,
                    not self.root.color,
                    False,
                    self.board.serialize_position()
                )
                self.board.pop()

                self.finished = True
                continue

            # 0 children, so nothing sent to evaluation
            try:
                node = self.traverser.get_node(item.control_value)
            except KeyError:
                print("control node not found")
                pass
                # TODO: should raise?
            else:
                # TODO: if no children then control if the score is valid for checkmate or stalemate
                node.parent.propagate_score(node.score, None, node.color, node.level)

    def _read_control_values(self) -> None:
        """"""

        self.distributed = self.memory_manager.get_int(DISTRIBUTED)

    def _wait_closed_status(self) -> None:
        """"""

        while True:
            current_status = self.memory_manager.get_int(STATUS)
            if current_status == CLOSED:
                return
            sleep(SLEEP)

        # print("distributor non-responsive")
        # raise SearchFailed

    def handle_candidates(
        self, distributor_queue: QM, candidates: List[SelectorItem]
    ) -> None:
        """"""

        nodes_to_distribute: List[Node] = self.traverser.get_nodes_to_distribute(
            candidates
        )

        if nodes_to_distribute:
            self.distributed += len(nodes_to_distribute)
            self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)

            self.explored += len(nodes_to_distribute)
            self.queue_for_distribution(
                distributor_queue, nodes_to_distribute, recaptures=True
            )

    def select_from_tree(self, distributor_queue: QM, iterations: int = 1) -> None:
        """
        Select next nodes to explore and queue for distribution.
        """

        top_nodes = self.traverser.get_nodes_to_look_at(iterations)
        if not top_nodes:
            return

        n_nodes: int = len(top_nodes)
        self.selected += n_nodes
        self.explored += n_nodes
        self.distributed += n_nodes

        self.queue_for_distribution(distributor_queue, top_nodes, recaptures=False)

    def queue_for_distribution(
        self, distributor_queue: QM, nodes: List[Node], *, recaptures: bool
    ) -> None:
        """"""

        # distributor_queue.put_many(
        #     [
        #         DistributorItem(
        #             self.run_id,
        #             node.name,
        #             node.move,
        #             node.captured if recaptures else -1 if node.only_captures else 0,
        #             node.score,
        #             node.board
        #         )
        #         for node in nodes
        #     ]
        # )

        to_queue: List[DistributorItem] = []
        for node in nodes:
            captured: int
            if recaptures:
                captured = node.captured
            elif node.only_captures:
                captured = -1
                node.only_captures = False
            else:
                captured = 0

            to_queue.append(DistributorItem(
                self.run_id,
                node.name,
                node.move,
                captured,
                node.score,
                node.board
            ))

        distributor_queue.put_many(to_queue)

    async def harvesting_loop(
        self,
        selector_queue: QM,
        distributor_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        while True:
            self.harvest_queues(
                selector_queue, distributor_queue, control_queue, queue_throttle
            )
            await asyncio_sleep(0)

    async def selecting_loop(self, distributor_queue: QM) -> None:
        """"""

        while True:
            self.select_from_tree(distributor_queue)
            await asyncio_sleep(0)

    async def monitoring_loop(self):
        """"""

        while True:
            if self.monitor():
                break
            await asyncio_sleep(0)

    def finish_up(self, total_time: float) -> str:
        """"""

        if self.printing == Print.TREE:
            min_depth, max_depth, path = self.tree_params.split(",")
            self.print_tree(int(min_depth), int(max_depth), path)
            # print(PrunedTreeRenderer(self.root, depth=3, maxlevel=3))

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
            if child.leaf_level >= 2/3 * depth:
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
