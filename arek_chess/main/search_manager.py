"""
Handling of the search tree including following features: depth control, branch pruning, printing output
"""

import time
from asyncio import sleep as asyncio_sleep
from typing import Optional, Dict, List, Tuple

from arek_chess.board.board import Board
from arek_chess.constants import (
    Print,
    INF,
    ROOT_NODE_NAME,
    SLEEP,
    LOG_INTERVAL,
)
from arek_chess.main.game_tree.node.node import Node
from arek_chess.main.game_tree.renderer import PrunedTreeRenderer
from arek_chess.main.game_tree.traversal import Traversal
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager


class SearchManager:
    """
    Class_docstring
    """

    tree: Dict[str, Node]

    def __init__(
        self,
        dispatcher_queue: QueueManager,
        eval_queue: QueueManager,
        candidates_queue: QueueManager,
        printing: Print,
        tree_params: str,
    ):
        super().__init__()

        self.dispatcher_queue = dispatcher_queue
        self.eval_queue = eval_queue
        self.candidates_queue = candidates_queue

        self.printing = printing
        self.tree_params = tree_params  # TODO: create a constant class like Print

        self.memory_manager = MemoryManager()

        self.dispatched = 0
        self.evaluated = 0
        self.selected = 0

    def set_root(self, fen: Optional[str] = None, turn: Optional[bool] = None) -> None:
        """"""

        if fen:
            board = Board(fen)
            if turn is not None:
                board.turn = turn
        else:
            board = Board()

        score = -INF if board.turn else INF

        self.dispatcher_queue.put((ROOT_NODE_NAME, "", score))

        self.root = Node(
            parent=None,
            name=ROOT_NODE_NAME,
            move="",
            score=score,
            captured=0,
            level=0,
            color=board.turn,
        )
        self.root.looked_at = True
        self.memory_manager.set_node_board(ROOT_NODE_NAME, board)

        self.nodes_dict = {ROOT_NODE_NAME: self.root}

        self.traversal = Traversal(self.root)

    def search(self) -> str:
        """"""

        self.limit: int = 2**14  # 14 -> 16384, 15 -> 32768

        self.t_0: float = time.time()
        self.t_tmp: float = self.t_0
        self.last_evaluated: int = 0

        max_at_once = self.limit / 16  # arbitrarily, so that this process doesn't drag

        try:
            while self.evaluated < self.limit:  # TODO: stop when queue is empty
                # TODO: optimize concurrency
                self.gathering(max_at_once)
                self.selecting()
                # if self.timing():
                #     break

        except KeyboardInterrupt:
            self.before_exit()
        else:
            return self.finish_up(time.time() - self.t_0)

    def timing(self) -> True:
        """"""

        t = time.time()

        if t > self.t_tmp + LOG_INTERVAL:
            self.t_tmp = t

            progress = round(
                min(
                    self.evaluated / self.dispatched,
                    self.evaluated / self.limit,
                )
                * 100
            )
            print(
                f"evaluated: {self.evaluated}, dispatched: {self.dispatched}, progress: {progress}%"
            )

            if self.evaluated == self.last_evaluated:
                return True

            self.last_evaluated = self.evaluated

        return False

    async def timing_loop(self):
        """"""

        while True:
            if self.timing():
                break
            await asyncio_sleep(0)

    def gathering(self, max_at_once):
        """"""

        candidates = self.candidates_queue.get_many(max_at_once)  # self.max_items_at_once)
        if not candidates:
            time.sleep(SLEEP)
            # print("no items")
            return

        self.evaluated += len(candidates)
        self.handle_candidates(candidates)

    async def gathering_loop(self, max_at_once: int) -> None:
        """"""

        while True:
            self.gathering(max_at_once)
            await asyncio_sleep(0)

    def handle_candidates(self, candidates):
        """"""

        nodes_to_dispatch = []

        # best_score = math.inf if color else -math.inf
        for candidate in candidates:
            (
                parent_name,
                move_str,
                moved_piece_type,
                captured_piece_type,
                is_check,
                score,
            ) = candidate
            parent = self.get_node(parent_name)
            level = len(parent_name.split("."))

            to_dispatch = False
            node: Node = self.create_node(
                parent, move_str, score, captured_piece_type, level
            )

            # analyse capture-fest immediately, provided at least equal value of piece captured as previously
            if captured_piece_type:
                piece_value = Board.get_simple_piece_value(moved_piece_type)
                capture_value = Board.get_simple_piece_value(captured_piece_type)
                parent_capture_value = Board.get_simple_piece_value(parent.captured)

                if capture_value > parent_capture_value or (
                    piece_value > parent_capture_value
                    and capture_value == parent_capture_value
                ):
                    nodes_to_dispatch.append(node)
                    node.being_processed = True
                    self.dispatched += 1
                    # to_dispatch = True
            if not to_dispatch:
                parent.being_processed = False

        if nodes_to_dispatch:
            self.queue_to_dispatch(nodes_to_dispatch)

    def selecting(self) -> None:
        """"""

        if (
            self.dispatched > 0 and self.evaluated / self.dispatched < 0.75
        ) or self.evaluated > self.limit:
            return

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
        self.dispatched += n_nodes

        self.queue_to_dispatch(top_nodes)

    def queue_to_dispatch(self, nodes: List[Node]) -> None:
        """"""

        self.dispatcher_queue.put_many(
            [(node.name, node.move, node.score) for node in nodes]
        )

    async def selecting_loop(self):
        """"""

        while True:
            self.selecting()
            await asyncio_sleep(0)

    def get_node(self, node_name: str) -> Node:
        """"""

        return self.nodes_dict[node_name]

    def create_node(self, parent: Node, move, score, captured, level) -> Node:
        """"""

        parent_name = parent.name
        child_name = f"{parent_name}.{move}"
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
            print(PrunedTreeRenderer(self.root, depth=int(min_depth), maxlevel=int(max_depth), path=path))
            # print(PrunedTreeRenderer(self.root, depth=3, maxlevel=3))

        best_score, best_move = self.get_best_move()

        if self.printing != Print.NOTHING:
            print(f"evaluated: {self.evaluated}, dispatched: {self.dispatched}, selected: {self.selected}")

            print("chosen move -->", round(best_score, 3), best_move)

            print(f"time: {total_time}, nodes/s: {round(self.evaluated / total_time)}")

        self.evaluated = 0
        self.selected = 0
        self.dispatched = 0

        return best_move

    def get_best_move(self) -> Tuple[float, str]:
        """
        Get the first move with the highest score with respect to color.
        """

        sorted_children: List[Node] = sorted(
            self.root.children, key=lambda node: node.score, reverse=self.root.color
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
            print(PrunedTreeRenderer(self.root, depth=int(min_depth), maxlevel=int(max_depth), path=path))
            print(f"evaluated: {self.evaluated}, dispatched: {self.dispatched}, selected: {self.selected}")
