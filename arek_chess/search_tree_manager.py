# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import math
import time
import traceback
from statistics import mean

from anytree import Node, RenderTree, LevelOrderIter
from anytree.render import _is_last

from arek_chess.board.board import Board
from arek_chess.common_data_manager import CommonDataManager
from arek_chess.constants import DEPTH
from arek_chess.dispatcher import Dispatcher
from arek_chess.messaging import Queue

NOTHING = 0
FIRST_LEVEL = 1
TREE = 2

PRINT = NOTHING


class PrintableNode(Node):
    def __repr__(self):
        return f"Node({self.level}, {self.move}, {self.score})"


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
                for grandchild in self.__next(child, continues + (not is_last, ), level=level):
                    yield grandchild

    def has_living_family(self, node: Node):
        if node.level >= DEPTH:
            return True

        for i in node.children:
            if self.has_living_family(i):
                return True

        return False


class SearchTreeManager:
    """
    Class_docstring
    """

    SLEEP = 0.001
    ROOT_NAME = "0"

    def __init__(self, fen: str, turn: bool, depth: int = DEPTH):
        self.turn = turn
        self.depth = depth

        self.root = PrintableNode(self.ROOT_NAME, turn=turn, score=0, fen=fen, level=0, move=None)
        self.tree = {self.ROOT_NAME: self.root}

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")
        self.to_erase_queue = Queue("candidate")

        CommonDataManager.set_node_board(self.ROOT_NAME, Board())

    def run_search(self):
        dispatcher = Dispatcher(self.node_queue, self.candidates_queue, self.to_erase_queue)
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

        self.node_queue.put((self.root.name, self.root.fen, self.turn))

        sent = 1
        returned = 0
        dominated = 0

        t0 = time.time()
        while sent > returned:
            t = time.time()
            if t > t0 + 10:
                print(sent, returned, dominated)
                t0 = t

            candidates = self.candidates_queue.get()
            if not candidates:
                time.sleep(self.SLEEP)
                continue

            parent = self.get_node(candidates[0]["node_name"])
            parent_name_split = parent.name.split(".")
            level = len(parent_name_split)
            turn_after = level % 2 == 0 if self.turn else level % 2 == 1

            for candidate in candidates:
                node, is_dominated = self.create_node(candidate, parent, level, turn_after)
                if is_dominated:
                    dominated += 1
                    self.to_erase_queue.put(node.name)
                    continue

                if node.level < self.depth or node.is_capture:
                    sent += 1
                    self.node_queue.put((node.name, node.fen, turn_after))

            returned += 1

        best_move = self.get_best_move()

        if PRINT == TREE:
            print(PrunedRenderTree(self.root))

        print(sent, returned, dominated)

        print(best_move)

    def get_node(self, node_name):
        return self.tree[node_name]

    def create_node(self, candidate, parent, level, turn_after):
        parent_name = parent.name
        child_name = f"{parent_name}.{candidate['i']}"

        score = candidate["score"]
        move = candidate["move"]
        is_capture = candidate["is_capture"]
        fen = candidate["fen"]

        node = PrintableNode(
            child_name,
            parent=parent,
            score=round(score, 3),
            move=move,
            is_capture=is_capture,
            fen=fen,
            level=level,
        )

        self.tree[child_name] = node

        is_dominated = self.is_node_dominated(score, parent, turn_after) if level >= 4 and level < self.depth else False

        return node, is_dominated

    def is_node_dominated(self, score, parent: Node, color: bool):
        return self.is_below_average(score, parent, color) or self.is_not_promising(score, parent, color)

    def is_below_average(self, score, parent: Node, color: bool):
        grand_parents = parent.parent.parent.children
        avg = mean([node.score for node in grand_parents])
        if color:
            return score < avg
        else:
            return score > avg

    def is_not_promising(self, score, parent: Node, color: bool):
        try:
            trend = self.get_trend(score, parent)
        except KeyError as e:
            print(f"missing node: {e}")
            print(f'analysing for child of: {parent.name}')
            return False

        if color:
            # was growing, but rapidly decreasing growth
            if all([delta > 0 for delta in trend]) and trend[2] / trend[1] > trend[1] / trend[0]:
                return True

            # kept getting worse until dropped below 0
            if trend[0] < trend[1] < trend[2] and trend[0] < 0:
                return True
        else:
            # was growing, but rapidly decreasing growth
            if all([delta < 0 for delta in trend]) and trend[2] / trend[1] > trend[1] / trend[0]:
                return True

            # kept getting worse until dropped below 0
            if trend[0] > trend[1] > trend[2] and trend[0] > 0:
                return True

        return False

    def get_trend(self, score, parent: Node):
        """recent go first"""
        consecutive_scores = [score, *self.get_consecutive_scores(parent)]
        averages = [(consecutive_scores[i] + consecutive_scores[i+1] / 2) for i in range(4)]
        deltas = [averages[i + 1] - averages[i] for i in range(3)]
        return deltas

    def get_consecutive_scores(self, parent: Node):
        return [parent.score, parent.parent.score, parent.parent.parent.score, parent.parent.parent.parent.score]

    def get_best_move(self):
        parent = None
        node: Node

        # walk over all leafs reversed (from furthest children) and backprop scores on parents
        ordered_nodes = [node for node in LevelOrderIter(self.root)]
        for node in reversed(ordered_nodes):
            if node.level < 1:
                break

            if node.level < DEPTH and not hasattr(node, "has_leaf"):
                continue

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

            if PRINT == FIRST_LEVEL and node.level == 1:
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
