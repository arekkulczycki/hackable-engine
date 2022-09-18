"""
Handling of the search tree including following features: depth control, branch pruning, printing output
"""

import asyncio
import time
from bisect import insort
from statistics import mean, stdev, median
from typing import Optional

from anytree import Node, LevelOrderIter

from arek_chess.board.board import Board
from arek_chess.main.dispatcher import Dispatcher
from arek_chess.main.game_tree.constants import Print, INF, ROOT_NODE_NAME
from arek_chess.main.game_tree.helpers import GetBestMoveMixin
from arek_chess.main.game_tree.node import BackpropNode
from arek_chess.main.game_tree.renderer import PrunedTreeRenderer
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.messaging import Queue

LOG_INTERVAL = 1

CPU_CORE_BUFFER = 100

COMPARISON_DEPTH = 4

TREE_MAX_LEVEL = 999

# remove_shm_from_resource_tracker()


class SearchManager(GetBestMoveMixin):
    """
    Class_docstring
    """

    SLEEP = 0.001
    PRINT: int = Print.NOTHING

    def __init__(
        self,
        eval_queue: Queue,
        candidates_queue: Queue,
        cpu_cores: int,
        depth: int,
        width: float = 0.001,
    ):
        self.denominator = int(1 / width)
        self.cpu_cores = cpu_cores
        self.depth = depth
        self.width = width
        self.eval_queue = eval_queue
        self.candidates_queue = candidates_queue

        self.memory_manager = MemoryManager()
        self.dispatcher = Dispatcher(self.eval_queue, self.candidates_queue)

        # self.pruner = ArekPruner()

    def set_root(self, fen: Optional[str] = None, turn: Optional[bool] = None) -> None:
        """"""

        if fen:
            board = Board(fen)
            if turn is not None:
                board.turn = turn
                self.root_turn = turn
            else:
                self.root_turn = board.turn
        else:
            board = Board()
            self.root_turn = True

        self.turn = board.turn
        score = -INF if self.turn else INF

        self.root = BackpropNode(
            ROOT_NODE_NAME,
            score=score,
            init_score=score,
            level=0,
            color=self.turn,
            move=None,
            # pruned=False,
        )
        self.memory_manager.set_node_board(ROOT_NODE_NAME, board)

        self.tree = {ROOT_NODE_NAME: self.root}
        self.tree_stats = {}

        self.sorted_nodes = []

    def search(self):
        """"""

        self.initiated = 0
        self.calculated = 0
        self.queued = 0

        loop = asyncio.get_event_loop() or asyncio.new_event_loop()

        async_results, _ = loop.run_until_complete(
            asyncio.wait(
                [self.dispatching(), self.sorting()],
                return_when=asyncio.FIRST_COMPLETED,
            )
        )

        loop.close()

        return next(iter(async_results)).result()

    async def dispatching(self) -> str:
        """"""

        self.dispatcher.dispatch(self.root)
        self.initiated += 1

        t0 = time.time()
        t1 = t0
        last_calculated = 0
        n_sorted_nodes = 0
        root_level = True
        while self.initiated > self.calculated:
            t = time.time()
            # n_sorted_nodes = len(self.sorted_nodes)
            if t > t1 + LOG_INTERVAL:
                t1 = t
                print(f"initiated: {self.initiated}, candidates: {self.queued}, calculated: {self.calculated}")

                if self.calculated == last_calculated or self.calculated > 1000:
                    break

                last_calculated = self.calculated

            candidates_item = self.candidates_queue.get()
            if not candidates_item:
                await asyncio.sleep(0)
                # print("no items")
                continue

            self.handle_candidates(candidates_item, root_level)
            root_level = False

            self.calculated += 1

            await asyncio.sleep(0)

        return self.finish_up(n_sorted_nodes, t0)

    def handle_candidates(self, candidates_item, root_level=False):
        """"""

        parent_name, candidates = candidates_item
        parent = self.get_node(parent_name)
        parent_name_split = parent_name.split(".")
        level = len(parent_name_split)
        color = level % 2 == 0 if self.turn else level % 2 == 1

        nodes_to_dispatch = []

        # best_score = math.inf if color else -math.inf
        for candidate in candidates:
            self.initiated += 1

            node: Node = self.create_node(candidate, parent, level, color)
            # if score > best_score and not color or score < best_score and color:
            #     best_score = score

            if root_level:
                node.first_ancestor = node.move
            else:
                node.first_ancestor = parent.first_ancestor

            # always analyse capture-fest until the last capture
            if node.captured or root_level:  # or node.level % 2 != 1
                nodes_to_dispatch.append(node)
                # self.dispatcher.dispatch(node)
            else:
                # the sorting is done only for the positions when it's same player turn
                insort(self.sorted_nodes, node)
                self.queued += 1

        if nodes_to_dispatch:
            self.dispatcher.dispatch_many(nodes_to_dispatch)

    def finish_up(self, n_sorted_nodes, t0) -> str:
        """"""

        best_score, best_move = self.get_best_move(
            self.root, self.depth, self.root_turn
        )

        execution_time = time.time() - t0

        if self.PRINT == Print.TREE:
            print(PrunedTreeRenderer(self.depth, self.root, maxlevel=TREE_MAX_LEVEL))

        # for level, data in self.tree_stats.items():
        #     print(f"median for level {level}: {data.get('median')}")

        print(
            f"initiated: {self.initiated}, scheduled: {self.queued}, calculated: {self.calculated}"
        )

        print("chosen move -->", best_score, best_move)

        print(f"time: {execution_time}")

        return best_move

    async def sorting(self):
        """"""

        last_printed_top_choice = None
        last_top_choice = None
        dispatched = 0

        while True:
            await asyncio.sleep(0)

            n_to_queue = 8

            top_nodes = self.sorted_nodes[:n_to_queue]
            if top_nodes:
                del self.sorted_nodes[:n_to_queue]

                last_top_choice = top_nodes[0].first_ancestor

                for node in top_nodes:
                    self.dispatcher.dispatch(node)

                dispatched += 8

            if last_top_choice != last_printed_top_choice:
                last_printed_top_choice = last_top_choice
                print(f"top choice: {last_top_choice}")

            # n_candidates = len(self.sorted_nodes)
            # if n_candidates > 10000:
            #     del self.sorted_nodes[n_candidates//2:]

    def get_node(self, node_name):
        """"""

        return self.tree[node_name]

    def create_node(self, candidate, parent, level, color) -> Node:
        """"""

        move = candidate["move"]
        captured = candidate["captured"]
        score = candidate["score"]

        parent_name = parent.name
        child_name = f"{parent_name}.{move}"

        node = BackpropNode(
            child_name,
            parent=parent,
            score=score,
            init_score=score,
            move=move,
            captured=captured,
            level=level,
            color=color,
            # pruned=False,
        )

        self.tree[child_name] = node

        # to_prune = self.pruner.should_prune(
        #     self.tree_stats, score, parent, not turn_after, captured, self.depth
        # )
        # node.pruned = to_prune

        return node  # , to_prune

    def gather_level_stats(self, level: int) -> None:
        """"""

        if level in self.tree_stats:
            return

        nodes = []
        scores = []
        for node in LevelOrderIter(self.root, maxlevel=level + 1):
            if node.level != level:
                continue

            nodes.append(node)
            scores.append(node.score)

        _number = len(nodes)
        # number = self.tree_stats.get(level, {}).get("number")
        #
        # if number != _number:
        _min = min(scores)
        _max = max(scores)
        _median = median(scores)
        _mean = mean(scores) if _number > 3 else None
        _stdev = stdev(scores) if _number > 3 else None

        self.tree_stats[level] = {
            "number": _number,
            "min": _min,
            "max": _max,
            "median": _median,
            "mean": _mean,
            "stdev": _stdev,
        }

    @classmethod
    def run_clean(cls, depth, width):
        """"""

        cls.clean_children("0", 0, depth, width)

    @classmethod
    def clean_children(cls, node_name, level, depth, width):
        """"""

        for i in range(width):
            if level <= depth:
                cls.clean_children(f"{node_name}.{i}", level + 1, depth, width)
            try:
                print(f"cleaning {node_name}...")
                memory_manager = MemoryManager()
                memory_manager.remove_node_params_memory(node_name)
                memory_manager.remove_node_board_memory(node_name)
            except:
                continue
