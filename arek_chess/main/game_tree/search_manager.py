"""
Handling of the search tree including following features: depth control, branch pruning, printing output
"""

import asyncio
import time
from typing import Optional, Dict

from arek_chess.board.board import Board
from arek_chess.main.dispatcher import Dispatcher
from arek_chess.main.game_tree.constants import Print, INF, ROOT_NODE_NAME
from arek_chess.main.game_tree.helpers import GetBestMoveMixin
from arek_chess.main.game_tree.node.node import Node
from arek_chess.main.game_tree.node.create_node_mixin import CreateNodeMixin
from arek_chess.main.game_tree.renderer import PrunedTreeRenderer
from arek_chess.main.game_tree.traversal import Traversal
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager

LOG_INTERVAL = 1

# remove_shm_from_resource_tracker()


class SearchManager(GetBestMoveMixin, CreateNodeMixin):
    """
    Class_docstring
    """

    SLEEP = 0.0001
    PRINT: int = Print.NOTHING

    tree: Dict[str, Node]

    def __init__(
        self,
        eval_queue: QueueManager,
        candidates_queue: QueueManager,
    ):
        super().__init__()

        self.eval_queue = eval_queue
        self.candidates_queue = candidates_queue

        self.memory_manager = MemoryManager()
        self.dispatcher = Dispatcher(self.eval_queue)

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

        self.root = Node(
            parent=None,
            name=ROOT_NODE_NAME,
            move="",
            score=score,
            captured=0,
            level=0,
            color=board.turn,
        )
        self.memory_manager.set_node_board(ROOT_NODE_NAME, board)

        self.tree = {ROOT_NODE_NAME: self.root}

        self.traversal = Traversal(self.root, self.tree)

    def search(self):
        """"""

        self.limit = 10000

        self.t0 = time.time()
        self.t_tmp = self.t0
        self.last_evaluated = 0

        self.dispatcher.dispatch(self.root)

        try:
            while (
                self.evaluated < self.dispatcher.dispatched
                or self.evaluated < self.limit
            ):
                # TODO: optimize concurrency
                self.dispatching()
                self.selecting()
                if self.timing():
                    break

        except KeyboardInterrupt:
            self.before_exit()
        else:
            return self.finish_up(time.time() - self.t0)

    def timing(self) -> True:
        """"""

        t = time.time()

        if t > self.t_tmp + LOG_INTERVAL:
            self.t_tmp = t

            progress = round(
                min(
                    self.evaluated / self.dispatcher.dispatched,
                    self.evaluated / self.limit,
                )
                * 100
            )
            print(
                f"evaluated: {self.evaluated}, dispatched: {self.dispatcher.dispatched}, progress: {progress}%"
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
            await asyncio.sleep(0)

    def dispatching(self):
        """"""

        candidates = self.candidates_queue.get_many(1024)  # self.max_items_at_once)
        if not candidates:
            time.sleep(self.SLEEP)
            # print("no items")
            return

        self.evaluated += len(candidates)
        self.handle_candidates(candidates)

    async def dispatching_loop(self):
        """"""

        while True:
            self.dispatching()
            await asyncio.sleep(0)

    def handle_candidates(self, candidates):
        """"""

        nodes_to_dispatch = []

        # best_score = math.inf if color else -math.inf
        for candidate in candidates:
            (
                parent_name,
                size,
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
                    to_dispatch = True
            if not to_dispatch:
                parent.being_processed = False

        if nodes_to_dispatch:
            self.dispatcher.dispatch_many(nodes_to_dispatch)

    def selecting(self) -> None:
        """"""

        if (
            self.evaluated / self.dispatcher.dispatched < 0.9
            or self.dispatcher.dispatched > self.limit
        ):
            return

        top_nodes = [
            node
            for node in (self.traversal.get_next_node_to_look_at() for _ in range(4))
            if node is not None
        ]
        if not top_nodes:
            # print("no nodes")
            return

        else:
            # print("selecting")
            self.selected += len(top_nodes)
            self.dispatcher.dispatch_many(top_nodes)

    async def selecting_loop(self):
        """"""

        while True:
            self.selecting()
            await asyncio.sleep(0)

    def get_node(self, node_name):
        """"""

        return self.tree[node_name]

    def finish_up(self, total_time: float) -> str:
        """"""

        # best_score, best_move = self.get_best_move(
        #     self.root, self.depth, self.root_turn
        # )
        best_score, best_move = self.get_best_move()

        if self.PRINT == Print.TREE:
            # print(PrunedTreeRenderer(self.root, depth=3, maxlevel=5))
            print(PrunedTreeRenderer(self.root, depth=5, maxlevel=5, path="f3e5"))
        elif self.PRINT == Print.CANDIDATES:
            print(PrunedTreeRenderer(self.root, depth=3, maxlevel=2))

        # print(f"evaluated: {self.evaluated}, dispatched: {self.dispatcher.dispatched}, selected: {self.selected}")

        print("chosen move -->", round(best_score, 3), best_move)

        print(f"time: {total_time}, nodes/s: {round(self.evaluated / total_time)}")

        self.evaluated = 0
        self.selected = 0
        self.dispatcher.dispatched = 0

        return best_move

    def before_exit(self, *args):
        """"""

        print(PrunedTreeRenderer(self.root, depth=6, maxlevel=7))
        print(f"evaluated: {self.evaluated}, dispatched: {self.dispatcher.dispatched}")

    # @classmethod
    # def run_clean(cls, depth, width):
    #     """"""
    #
    #     cls.clean_children("0", 0, depth, width)
    #
    # @classmethod
    # def clean_children(cls, node_name, level, depth, width):
    #     """"""
    #
    #     for i in range(width):
    #         if level <= depth:
    #             cls.clean_children(f"{node_name}.{i}", level + 1, depth, width)
    #         try:
    #             print(f"cleaning {node_name}...")
    #             memory_manager = MemoryManager()
    #             memory_manager.remove_node_params_memory(node_name)
    #             memory_manager.remove_node_board_memory(node_name)
    #         except:
    #             continue
