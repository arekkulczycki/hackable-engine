# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import math
import time
import traceback

from anytree import Node, RenderTree, find, LevelOrderIter

from arek_chess.dispatcher import Dispatcher
from arek_chess.messaging import Queue
from board.board import Board


class SearchTreeManager:
    """
    Class_docstring
    """

    SLEEP = 0.01

    def __init__(self, board: Board, turn: bool, depth: int = 8):
        self.root = board
        self.turn = turn
        self.depth = depth

        self.root = Node("0", turn=turn, score=0, fen=board.fen(), level=0)

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")

    def run_search(self):
        dispatcher = Dispatcher(self.node_queue, self.candidates_queue)
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

        self.node_queue.put((self.root.name, self.root.fen))

        sent = 1
        returned = 0

        while sent > returned:
            # if returned > pow(2, self.depth) - 100:
            #     print(returned)

            candidates = self.candidates_queue.get()
            if not candidates:
                time.sleep(self.SLEEP)
                continue

            for i, candidate in enumerate(candidates):
                node_name = candidate["node_name"].replace("/", "")
                level = len(node_name.split("."))

                node = Node(
                    f"{node_name}.{i}",
                    parent=self.get_node(node_name),
                    score=candidate["score"],
                    move=candidate["move"],
                    fen=candidate["fen"],
                    level=level,
                )

                if level < self.depth:
                    sent += 1
                    self.node_queue.put((node.name, node.fen))

            returned += 1

        # print(RenderTree(self.root))

        print(self.get_best_move())

    def get_node(self, node_id):
        return find(self.root, lambda node: node.name == node_id)

    def get_best_move(self):
        parent = None
        node: Node

        # walk over all leafs reversed (from furthest children) and backprop scores on parents
        ordered_nodes = [node for node in LevelOrderIter(self.root)]
        for node in reversed(ordered_nodes):
            if node.level <= 1:
                break

            if node.parent != parent:
                parent = node.parent
                parent.score = node.score
            else:
                if (node.level % 2 == 1 and self.turn) or (node.level % 2 == 0 and not self.turn):
                    if node.score > parent.score:
                        parent.score = node.score
                else:
                    if node.score < parent.score:
                        parent.score = node.score

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
