"""
Handling of the search tree including following features: depth control, branch pruning, printing output
"""

from asyncio import sleep as asyncio_sleep
from time import time
from typing import Optional, Dict, List, Tuple

from numpy import double

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    Print,
    INF,
    ROOT_NODE_NAME,
    LOG_INTERVAL,
    FINISHED,
)
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.queue_manager import QueueManager as QM
from arek_chess.game_tree.node.node import Node
from arek_chess.game_tree.renderer import PrunedTreeRenderer
from arek_chess.game_tree.traversal import Traversal


class SearchManager:
    """
    Class_docstring
    """

    tree: Dict[str, Node]

    def __init__(
        self,
        dispatcher_queue: QM,
        selector_queue: QM,
        control_queue: QM,
        queue_throttle: int,
        printing: Print,
        tree_params: str,
        search_limit: Optional[int],
    ):
        super().__init__()

        self.dispatcher_queue = dispatcher_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = queue_throttle

        self.printing = printing
        self.tree_params = tree_params  # TODO: create a constant class like Print

        self.memory_manager = MemoryManager()

        self.dispatched = 0
        self.evaluated = 0
        self.selected = 0
        # self.selected_nodes = []
        self.explored = 0

        self.limit: int = 2 ** (search_limit or 14)  # 14 -> 16384, 15 -> 32768

    def set_root(self, board: Optional[Board] = None) -> None:
        """"""

        if board:
            self.board = board
        else:
            self.board = Board()

        score = -INF if self.board.turn else INF

        self.root = Node(
            parent=None,
            name=ROOT_NODE_NAME,
            move="",
            score=score,
            captured=0,
            level=0,
            color=self.board.turn,
        )
        self.root.looked_at = True

        self.nodes_dict = {ROOT_NODE_NAME: self.root}

        self.traversal = Traversal(self.root)

    def search(self) -> str:
        """"""

        self.memory_manager.set_node_board(ROOT_NODE_NAME, self.board)
        self.dispatcher_queue.put((ROOT_NODE_NAME, "", self.root.score))
        self.dispatched = 1

        self.t_0: float = time()
        self.t_tmp: float = self.t_0
        self.last_evaluated: int = 0

        limit = self.limit
        queue_throttle = self.queue_throttle
        dispatcher_queue = self.dispatcher_queue
        selector_queue = self.selector_queue
        control_queue = self.control_queue
        gathering = self.gathering
        selecting = self.selecting
        monitoring = self.monitoring
        try:
            while self.evaluated < self.dispatched or self.evaluated < self.limit:
                # TODO: refactor to use concurrency?
                gathering(
                    selector_queue, dispatcher_queue, control_queue, queue_throttle
                )
                if not (
                    self.dispatched == 0
                    or self.evaluated > limit
                    or self.evaluated / self.dispatched < 0.75
                    or (
                        self.evaluated / self.dispatched < 0.9
                        and self.evaluated < limit / 2
                    )
                ):
                    selecting(dispatcher_queue)

                if monitoring():
                    # print("no change detected")
                    raise SearchFailed("nodes not delivered on queues")

        except KeyboardInterrupt:
            self.before_exit()
        else:
            return self.finish_up(time() - self.t_0)
        finally:
            self.dispatcher_queue.put((FINISHED, "", 0.0))

    def monitoring(self) -> bool:
        """"""

        t = time()

        if t > self.t_tmp + LOG_INTERVAL:
            if self.evaluated == self.last_evaluated:
                print(
                    f"dispatched: {self.dispatched}, evaluated: {self.evaluated}, "
                    f"selected: {self.selected}"
                )
                return True

            self.t_tmp = t

            if self.printing not in [Print.NOTHING, Print.MOVE]:
                progress = round(
                    min(
                        self.evaluated / self.dispatched,
                        self.evaluated / self.limit,
                    )
                    * 100
                )

                print(
                    f"dispatched: {self.dispatched}, evaluated: {self.evaluated}, "
                    f"selected: {self.selected}, explored: {self.explored}, progress: {progress}%"
                )

            self.last_evaluated = self.evaluated

        return False

    async def monitoring_loop(self):
        """"""

        while True:
            if self.monitoring():
                break
            await asyncio_sleep(0)

    def gathering(
        self,
        selector_queue: QM,
        dispatcher_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        candidates: List[
            Tuple[str, str, int, int, double]
        ] = selector_queue.get_many_blocking(
            0.015,
            queue_throttle,  # the timeout value set for optimal first iteration start
        )  # self.max_items_at_once)
        if not candidates:
            # print(f"no items")
            return

        self.evaluated += len(candidates)

        self.handle_candidates(dispatcher_queue, candidates)

        control_values = control_queue.get_many(queue_throttle)
        if control_values:
            self.dispatched = control_values[-1]

    async def gathering_loop(
        self,
        selector_queue: QM,
        dispatcher_queue: QM,
        control_queue: QM,
        queue_throttle: int,
    ) -> None:
        """"""

        gathering = self.gathering
        while True:
            gathering(selector_queue, dispatcher_queue, control_queue, queue_throttle)
            await asyncio_sleep(0)

    def handle_candidates(
        self, dispatcher_queue: QM, candidates: List[Tuple[str, str, int, int, double]]
    ) -> None:
        """"""

        nodes_to_dispatch: List[Node] = []

        # best_score = math.inf if color else -math.inf
        for candidate in candidates:
            (
                parent_name,
                move_str,
                moved_piece_type,
                captured_piece_type,
                score,
            ) = candidate
            try:
                parent = self.get_node(parent_name)
                parent_score = parent.score
            except KeyError:
                # print("node not found in items: ", parent_name, self.nodes_dict.keys())
                if len(self.nodes_dict.keys()) == 1:
                    # was never meant to be here, but somehow queue delivers phantom items
                    continue
                else:
                    raise SearchFailed(
                        f"node not found in items: {parent_name}, {self.nodes_dict.keys()}"
                    )
            level = len(parent_name.split("."))

            node = self.create_node(parent, move_str, score, captured_piece_type, level)

            to_dispatch = False
            # analyse "good" captures immediately
            if captured_piece_type > 0 and (  # if parent_name == ROOT_NODE_NAME:
                (
                    parent.parent is None
                    and captured_piece_type > parent.captured
                    and not (captured_piece_type == 3 and parent.captured == 2)
                )
                or (
                    parent.parent is not None and score > parent.parent.score and score > parent.parent.init_score
                )
            ):
                nodes_to_dispatch.append(node)
                node.being_processed = True
                self.explored += 1
                to_dispatch = True
            if not to_dispatch:
                parent.being_processed = False
            parent.being_processed = False

        if nodes_to_dispatch:
            self.queue_to_dispatch(dispatcher_queue, nodes_to_dispatch)

    def selecting(self, dispatcher_queue: QM) -> None:
        """"""

        top_nodes = [
            node
            for node in (self.traversal.get_next_node_to_look_at() for _ in range(4))
            if node is not None
        ]
        if not top_nodes:
            # print("no nodes")
            return

        # print("selecting")
        n_nodes: int = len(top_nodes)
        self.selected += n_nodes
        self.explored += n_nodes

        self.queue_to_dispatch(dispatcher_queue, top_nodes)

    def queue_to_dispatch(self, dispatcher_queue: QM, nodes: List[Node]) -> None:
        """"""

        dispatcher_queue.put_many(
            [(node.name, node.move, node.score) for node in nodes]
        )

    async def selecting_loop(self, dispatcher_queue: QM) -> None:
        """"""

        while True:
            self.selecting(dispatcher_queue)
            await asyncio_sleep(0)

    def get_node(self, node_name: str) -> Node:
        """"""

        return self.nodes_dict[node_name]

    def create_node(
        self, parent: Node, move: str, score: double, captured: int, level: int
    ) -> Optional[Node]:
        """"""

        parent_name = parent.name
        child_name = f"{parent_name}.{move}"
        if captured == -1:  # node reversed from dispatcher when found checkmate
            try:
                self.nodes_dict[child_name].score = score
                self.nodes_dict[child_name].captured = -1
                return None
            except KeyError:
                # was never meant to be here, but somehow queue delivers phantom items
                pass

        color = level % 2 == 0 if self.root.color else level % 2 == 1

        node = Node(
            parent=parent,
            name=child_name,
            move=move,
            score=score,
            captured=captured,
            level=level,
            color=color,
        )

        self.nodes_dict[child_name] = node

        return node

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
                    f"evaluated: {self.evaluated}, dispatched: {self.dispatched}, "
                    f"selected: {self.selected}, explored: {self.explored}"
                    # f"selected: {','.join([node[0].name for node in self.selected_nodes])}, explored: {self.explored}"
                )

                print(
                    f"time: {total_time}, nodes/s: {round(self.evaluated / total_time)}"
                )

            print("chosen move -->", round(best_score, 3), best_move)
        elif self.printing == Print.MOVE:
            print(best_move)

        self.dispatched = 0
        self.evaluated = 0
        self.selected = 0
        self.explored = 0

        return best_move

    def get_best_move(self) -> Tuple[double, str]:
        """
        Get the first move with the highest score with respect to color.
        """

        explored_children = [
            child
            for child in self.root.children
            if child.children or child.captured == -1
        ]
        if not explored_children:
            raise SearchFailed("finished without analysis")

        sorted_children: List[Node] = sorted(
            explored_children, key=lambda node: node.score, reverse=self.root.color
        )

        if self.printing == Print.CANDIDATES:
            for child in sorted_children[:5]:
                print(child.move, child.score)

        best = sorted_children[0]
        return best.score, best.move

    def before_exit(self, *_):
        """"""

        if self.printing == Print.TREE:
            min_depth, max_depth, path = self.tree_params.split(",")
            print(
                PrunedTreeRenderer(
                    self.root, depth=int(min_depth), maxlevel=int(max_depth), path=path
                )
            )
            print(f"evaluated: {self.evaluated}, dispatched: {self.dispatched}")
