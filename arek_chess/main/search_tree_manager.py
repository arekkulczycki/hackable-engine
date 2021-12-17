# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import math
import time
import traceback
from typing import List

from anytree import Node, RenderTree, LevelOrderIter
from anytree.render import _is_last

from arek_chess.board.board import Board
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
        self.tree = {self.ROOT_NAME: self.root}

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")

        MemoryManager.set_node_board(self.ROOT_NAME, Board(fen))

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

        to_prune = (
            self.should_be_pruned(score, parent, not turn_after, captured)
            if level >= 4 and level < self.depth
            else False
        )

        return node, to_prune

    def should_be_pruned(self, score, parent: Node, color: bool, captured: int):
        not_promising = self.is_not_promising(score, parent, color)

        is_worse_than_last_generation = self.is_worse_than_last_generation(score, parent, color)

        return is_worse_than_last_generation or not_promising

    @staticmethod
    def is_worse_than_last_generation(score, parent: Node, color: bool):
        grand_parents = parent.parent.parent.children

        return (
            score < min([node.score for node in grand_parents])
            if color
            else score > max([node.score for node in grand_parents])
        )

    def is_not_promising(self, score, parent: Node, color: bool):
        try:
            trend = self.get_trend(score, parent)
        except KeyError as e:
            print(f"missing node: {e}")
            print(f"analysing for child of: {parent.name}")
            return False

        ret = False

        if color:
            # keeps falling and increased fall
            if all([delta < 0 for delta in trend]) and trend[0] < trend[1]:
                ret = True

            # kept getting worse until dropped below 0
            if (
                trend[0] < 0
                and trend[0] < trend[1] < trend[2]
                and (
                    trend[0] < -trend[1]
                    if trend[1] < 0
                    else (trend[0] + trend[1] < -trend[2])
                )
            ):
                ret = True
        else:  # black
            # keeps falling and increased fall
            if all([delta > 0 for delta in trend]) and trend[0] > trend[1]:
                ret = True

            # kept getting worse until dropped below 0
            if (
                trend[0] > 0
                and trend[0] > trend[1] > trend[2]
                and (trend[0] > -trend[1] or trend[0] + trend[1] > -trend[2])
            ):
                ret = True

        return ret

    def get_trend(self, score, parent: Node):
        """recent go first"""
        consecutive_scores = [score, *self.get_consecutive_scores(parent)]
        averages = [
            (consecutive_scores[i] + consecutive_scores[i + 1] / 2) for i in range(4)
        ]
        deltas = [averages[i + 1] - averages[i] for i in range(3)]
        return deltas

    @staticmethod
    def get_consecutive_scores(parent: Node):
        return [
            parent.score,
            parent.parent.score,
            parent.parent.parent.score,
            parent.parent.parent.parent.score,
        ]

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
        return True
        if node.level >= DEPTH or node.move == "checkmate":
            return True

        for i in node.children:
            if self.has_living_family(i):
                return True

        return False
