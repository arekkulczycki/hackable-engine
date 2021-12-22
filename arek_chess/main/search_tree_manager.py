# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import math
import time
import traceback
from statistics import mean, stdev, median
from typing import List, Dict

from anytree import Node, RenderTree, LevelOrderIter
from anytree.render import _is_last

from arek_chess.board.board import Board
from arek_chess.criteria.pruning.arek_pruner import ArekPruner
from arek_chess.main.controller import DEPTH
from arek_chess.main.dispatcher import Dispatcher
from arek_chess.utils.memory_manager import (
    MemoryManager,
    remove_shm_from_resource_tracker,
)
from arek_chess.utils.messaging import Queue

# material, safety, under_attack, mobility, king_mobility, king_threats
DEFAULT_ACTION: List[float] = [100.0, 1.0, -1.0, 1.0, -1.0, 2.0]

NOTHING = 0
CANDIDATES = 1
TREE = 2
tree_max_level = 7

PRINT: int = NOTHING

remove_shm_from_resource_tracker()


class SearchTreeManager:
    """
    Class_docstring
    """

    SLEEP = 0.001
    ROOT_NAME = "0"

    def __init__(
        self, fen: str, turn: bool, action: List[float] = None, depth: int = DEPTH
    ):
        self.action = action if action else DEFAULT_ACTION

        self.turn = turn
        self.depth = depth

        self.root = PrintableNode(
            self.ROOT_NAME, turn=turn, score=0, level=0, move=None
        )
        MemoryManager.set_node_board(self.ROOT_NAME, Board(fen))

        self.tree = {self.ROOT_NAME: self.root}
        self.tree_stats = {}

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")

        self.pruner = ArekPruner()

    def run_search(self):
        dispatcher = Dispatcher(self.node_queue, self.candidates_queue, self.action)
        dispatcher.start()

        try:
            self.search()
        except:
            traceback.print_exc()
        finally:
            dispatcher.stop()

    def search(self):
        """

        :return:
        """

        self.node_queue.put((self.root.name, self.turn))

        sent = 1
        returned = 0
        pruned = 0

        t0 = time.time()
        last_returned = 0
        while sent > returned:
            t = time.time()
            if t > t0 + 3:
                print(sent, returned, pruned)
                t0 = t
                if returned == last_returned and returned / sent > 0.99:
                    break
                last_returned = returned

            candidates_item = self.candidates_queue.get()
            if not candidates_item:
                time.sleep(self.SLEEP)
                continue

            parent_name, candidates = candidates_item
            parent = self.get_node(parent_name)
            parent_name_split = parent_name.split(".")
            level = len(parent_name_split)
            turn_after = level % 2 == 0 if self.turn else level % 2 == 1

            for candidate in candidates:
                node, to_prune = self.create_node(
                    candidate, parent, level, turn_after
                )
                if to_prune:
                    pruned += 1
                    continue

                if node.move != "checkmate" and (
                    node.level < self.depth or node.captured
                ):
                    sent += 1
                    self.node_queue.put((node.name, turn_after))

            returned += 1

        best_move = self.get_best_move()

        if PRINT == TREE:
            print(PrunedRenderTree(self.root, maxlevel=tree_max_level))

        print(sent, returned, pruned)

        print(best_move)

    def get_node(self, node_name):
        return self.tree[node_name]

    def create_node(self, candidate, parent, level, turn_after):
        parent_name = parent.name
        child_name = f"{parent_name}.{candidate['i']}"

        move = candidate["move"]
        captured = candidate["captured"]
        score = candidate["score"]

        node = PrintableNode(
            child_name,
            parent=parent,
            score=score,
            move=move,
            captured=captured,
            level=level,
        )

        self.tree[child_name] = node

        stats = self.tree_stats.get(level - 1)
        if not stats:
            stats = self.gather_level_stats(level - 1)
            self.tree_stats[level - 1] = stats

        to_prune = (
            self.pruner.should_prune(self.tree_stats, score, parent, not turn_after, captured, self.depth)
        )

        return node, to_prune

    def gather_level_stats(self, level: int) -> Dict:
        """"""

        nodes = []
        scores = []
        for node in LevelOrderIter(self.root, maxlevel=level + 1):
            if node.level != level:
                continue

            nodes.append(node)
            scores.append(node.score)

        _number = len(nodes)
        _min = min(scores)
        _max = max(scores)
        _median = median(scores)
        _mean = mean(scores) if _number > 3 else None
        _stdev = stdev(scores) if _number > 3 else None
        return {"number": _number, "min": _min, "max": _max, "median": _median, "mean": _mean, "stdev": _stdev}

    def get_best_move(self):
        parent = None
        node: Node

        # walk over all leafs reversed (from furthest children) and backprop scores on parents
        ordered_nodes = [node for node in LevelOrderIter(self.root)]
        for node in reversed(ordered_nodes):
            if node.level < 1:
                break

            # if (
            #     node.level < DEPTH
            #     and node.move != "checkmate"
            #     and not hasattr(node, "has_leaf")
            # ):
            #     continue

            if node.parent != parent:
                parent = node.parent
                parent.score = node.score
                parent.has_leaf = True
            else:
                color = node.level % 2 == 0 if self.turn else node.level % 2 == 1
                if not color:
                    if node.score > parent.score:
                        parent.score = node.score
                        parent.has_leaf = True
                else:
                    if node.score < parent.score:
                        parent.score = node.score
                        parent.has_leaf = True

            if PRINT == CANDIDATES and node.level == 1:
                print(node.move, node.score)

        # choose the best move among the 1st level candidates, with already scores backpropped from leaves
        score = -math.inf if self.turn else math.inf
        move = None
        for child in self.root.children:
            if self.turn:  # white
                if child.score > score:
                    move = child.move
                    score = child.score
            else:
                if child.score < score:
                    move = child.move
                    score = child.score

        return score, move


class PrintableNode(Node):
    def __repr__(self):
        return f"Node({self.level}, {self.move}, {self.score})"

    @property
    def _has_leaf(self):
        return getattr(self, "has_leaf", False)


class PrunedRenderTree(RenderTree):
    def __iter__(self):
        return self.__next(self.node, tuple())

    def __next(self, node, continues, level=0):
        yield self._RenderTree__item(node, continues, self.style)
        children = node.children
        new_children = ()
        for i in range(len(children)):
            child = children[i]
            if self.has_living_family(child):
                new_children += (child,)
        children = new_children

        level += 1
        if children and (self.maxlevel is None or level < self.maxlevel):
            children = self.childiter(children)
            for child, is_last in _is_last(children):
                for grandchild in self.__next(
                    child, continues + (not is_last,), level=level
                ):
                    yield grandchild

    def has_living_family(self, node: Node):
        if node.level >= DEPTH or node.move == "checkmate":
            return True

        for i in node.children:
            if self.has_living_family(i):
                return True

        return False
