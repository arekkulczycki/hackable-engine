# -*- coding: utf-8 -*-

import os
from asyncio import sleep as asyncio_sleep
from time import time
from typing import Optional, Dict, List, Tuple

from numpy import abs, float32

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    Print,
    INF,
    ROOT_NODE_NAME,
    LOG_INTERVAL,
    FINISHED, ERROR,
)
from arek_chess.common.custom_threads import ReturningThread
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.profiler_mixin import ProfilerMixin
from arek_chess.common.queue_manager import QueueManager as QM
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
        self.nodes_dict: Dict[str, Node] = {}
        self.memory_manager = MemoryManager()

        self.distributed: int = 0
        self.evaluated: int = 0
        self.selected: int = 0
        self.explored: int = 0

        self.limit: int = 2 ** (search_limit or 14)  # 14 -> 16384, 15 -> 32768

        self.distribution_finished: bool = False
        self.should_profile: bool = False
        self.debug: bool = False

    def set_root(
        self,
        board: Optional[Board] = None,
        root: Optional[Node] = None,
        nodes_dict: Optional[Dict[str, Node]] = None,
    ) -> None:
        """"""

        if board:
            self.board = board
        else:
            self.board = Board()

        score = -INF if self.board.turn else INF

        if root is None:
            self.root = Node(
                parent=None,
                name=ROOT_NODE_NAME,
                move="",
                score=score / 2,  # divided by 2 to differentiate from checkmate
                captured=0,
                level=0,
                color=self.board.turn,
                being_processed=True,
            )
            self.nodes_dict = {ROOT_NODE_NAME: self.root}
            self.root.looked_at = True

        else:
            self.root = root
            self.nodes_dict = nodes_dict

            # clean hashmap of discarded moves
            for key in list(self.nodes_dict.keys()):
                if not key.startswith(self.root.name):
                    del self.nodes_dict[key]

        self.memory_manager.set_node_board(self.root.name, self.board)

        if not self.root.children:
            self.distributor_queue.put((self.root.name, "", self.root.score, 0))

        self.traverser: Traverser = Traverser(self.root, self.nodes_dict)

    def run(self):
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

        self.t_0: float = time()
        self.t_tmp: float = self.t_0
        self.last_evaluated: int = 0

        limit = self.limit
        queue_throttle = self.queue_throttle
        distributor_queue = self.distributor_queue
        selector_queue = self.selector_queue
        control_queue = self.control_queue
        harvest_queues = self.harvest_queues
        distribute = self.distribute
        monitor = self.monitor

        try:
            # TODO: refactor to use concurrency?
            while not (self.distribution_finished and self.evaluated == self.distributed):
                harvest_queues(
                    selector_queue, distributor_queue, control_queue, queue_throttle
                )

                gap = 0
                ratio = 0
                if self.distributed != 0:
                    gap = self.distributed - self.evaluated
                    ratio = gap / self.distributed
                if self.distributed == 0 or (
                    self.distributed < limit
                    and gap < 25000
                    and not (ratio > 0.25 and gap > 10000)
                    and not (
                        ratio > 0.1
                        and gap > 10000
                        and self.distributed
                        < limit / 2  # slow down in the first half of the search
                    )
                ):
                    distribute(  # TODO: the param should depend on both gap and speed
                        distributor_queue,
                        1
                        if gap > 10000
                        else 2
                        if gap > 3000
                        else 3
                        if self.distributed > 10000
                        else 1,
                    )

                if monitor():
                    break

        finally:
            # msg to distributor so that it resets counters
            self.distributor_queue.put((FINISHED, "", 0.0, 0))

        return self.finish_up(time() - self.t_0)

    def monitor(self) -> bool:
        """
        Monitor search progress and log events.

        :return: if the process hang and should be finished
        """

        t = time()

        if t > self.t_tmp + LOG_INTERVAL:
            if self.printing not in [Print.NOTHING, Print.MOVE]:
                os.system("clear")

            progress = round(
                min(
                    self.evaluated / self.distributed,
                    self.evaluated / self.limit,
                )
                * 100
            ) if self.distributed > 0 else 0

            if self.evaluated == self.last_evaluated:
                if progress < 100 and abs(self.root.score) == INF:
                    print(
                        f"distributed: {self.distributed}, evaluated: {self.evaluated}, selected: {self.selected}"
                    )
                    # print(
                    #     PrunedTreeRenderer(
                    #         self.root, depth=0, maxlevel=2
                    #     )
                    # )
                    # self.debug = True
                    # return False
                    raise SearchFailed("nodes not delivered on queues")

                return True

            self.t_tmp = t

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
                    print(child.move, child.score)

            self.last_evaluated = self.evaluated

        return self._stop_event.is_set()

    async def monitoring_loop(self):
        """"""

        while True:
            if self.monitor():
                break
            await asyncio_sleep(0)

    def harvest_queues(
        self,
        selector_queue: QM,
        distributor_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        self._handle_control_queue(control_queue)

        candidates: List[
            Tuple[str, str, float32, int]
        ] = selector_queue.get_many(
            queue_throttle,
        )
        if not candidates:
            # print(f"no items")
            return

        self.evaluated += len(candidates)

        self.handle_candidates(distributor_queue, candidates)

    def _handle_control_queue(self, control_queue) -> None:
        """"""

        # value = control_queue.get()
        # if value:
        #     try:
        #         parsed_value = int(value)
        #     except ValueError:
        #         if value == ERROR:
        #             raise SearchFailed("Dispatcher error")
        #
        #         if value == FINISHED:
        #             self.distribution_finished = True
        #
        #         # no children to look at, so nothing sent to evaluation
        #         node = self.traverser.get_node(value)
        #         node.being_processed = False
        #         node.parent.propagate_score(node.score, None, node.color)
        #     else:
        #         self.distributed = parsed_value

        control_values = control_queue.get_many(25)
        if control_values:
            number_found = False
            for control_value in reversed(control_values):
                try:
                    parsed_value = int(control_value)
                except ValueError:
                    if control_value == ERROR:
                        raise SearchFailed("Dispatcher error")

                    if control_value == FINISHED:
                        self.distribution_finished = True
                        continue

                    # no children to look at, so nothing sent to evaluation
                    try:
                        node = self.traverser.get_node(control_value)
                    except KeyError:
                        pass
                        # TODO: should raise?
                    else:
                        node.being_processed = False
                        node.parent.propagate_score(node.score, None, node.color)
                else:
                    if number_found:
                        continue
                    else:
                        number_found = True
                        self.distributed = parsed_value

    async def harvesting_loop(
        self,
        selector_queue: QM,
        distributor_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        while True:
            self.harvest_queues(selector_queue, distributor_queue, control_queue, queue_throttle)
            await asyncio_sleep(0)

    def handle_candidates(
        self, distributor_queue: QM, candidates: List[Tuple[str, str, float32, int]]
    ) -> None:
        """"""

        nodes_to_distribute: List[Node] = self.traverser.get_nodes_to_distrubute(
            candidates
        )

        if nodes_to_distribute:
            self.explored += len(nodes_to_distribute)
            self.queue_for_distribution(
                distributor_queue, nodes_to_distribute, recaptures=True
            )

    def distribute(self, distributor_queue: QM, iterations: int = 1) -> None:
        """
        Select next nodes to explore and queue for distribution.
        """

        top_nodes = self.traverser.get_nodes_to_look_at(iterations, self.debug)

        if self.debug:
            print(top_nodes)
        if not top_nodes:
            return

        n_nodes: int = len(top_nodes)
        self.selected += n_nodes
        self.explored += n_nodes

        self.queue_for_distribution(distributor_queue, top_nodes, recaptures=False)

    def queue_for_distribution(
        self, distributor_queue: QM, nodes: List[Node], *, recaptures: bool
    ) -> None:
        """"""

        distributor_queue.put_many(
            [
                (node.name, node.move, node.score, node.captured if recaptures else 0)
                for node in nodes
            ]
        )

    async def selecting_loop(self, distributor_queue: QM) -> None:
        """"""

        while True:
            self.selecting(distributor_queue)
            await asyncio_sleep(0)

    def finish_up(self, total_time: float) -> str:
        """"""

        if self.printing == Print.TREE:
            min_depth, max_depth, path = self.tree_params.split(",")
            print(
                PrunedTreeRenderer(
                    self.root, depth=int(min_depth), maxlevel=int(max_depth), path=path
                )
            )
            # print(PrunedTreeRenderer(self.root, depth=3, maxlevel=3))

        best_score, best_move = self.get_best_move()

        if self.printing not in [Print.NOTHING, Print.MOVE]:
            if self.printing != Print.MOVE:
                print(
                    f"evaluated: {self.evaluated}, distributed: {self.distributed}, "
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
            os.system("clear")
            for child in sorted_children[:]:
                print(child.move, child.score)

        best = sorted_children[0]
        return best.score, best.move
