# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import math
import time
import traceback

from anytree import Node, RenderTree, LevelOrderIter
from chess import Move

from arek_chess.common_data_manager import CommonDataManager
from arek_chess.dispatcher import Dispatcher
from arek_chess.messaging import Queue
from board.board import Board


class SearchTreeManager:
    """
    Class_docstring
    """

    SLEEP = 0.001
    ROOT_NAME = "0"

    def __init__(self, fen: str, turn: bool, depth: int = 4):
        self.turn = turn
        self.depth = depth

        self.root = Node(self.ROOT_NAME, turn=turn, score=0, fen=fen, level=0)
        self.tree = {self.ROOT_NAME: self.root}

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")

        # flush for fresh score recalculation
        from keydb import KeyDB

        db = KeyDB(host="localhost")
        db.flushdb()

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

        self.node_queue.put((self.root.name, self.root.fen, self.turn))

        sent = 1
        returned = 0

        t0 = time.time()
        while sent > returned:
            t = time.time()
            if t > t0 + 10:
                print(sent, returned)
                t0 = t

            candidates = self.candidates_queue.get()
            if not candidates:
                time.sleep(self.SLEEP)
                continue

            for i, candidate in enumerate(candidates):
                node = self.create_node(i, candidate)

                turn_after = node.level % 2 == 0 if self.turn else node.level % 2 == 1

                if node.level < self.depth:
                    sent += 1
                    self.node_queue.put((node.name, node.fen, turn_after))

            returned += 1

        print(RenderTree(self.root))

        print(sent, returned)

        print(self.get_best_move())

    def get_node(self, node_name):
        return self.tree[node_name]

    def create_node(self, i, candidate):
        parent_name = candidate["node_name"].replace("/", "")
        parent_name_split = parent_name.split(".")
        level = len(parent_name_split)
        child_name = f"{parent_name}.{i}"

        node = Node(
            child_name,
            parent=self.get_node(parent_name),
            score=round(candidate["score"], 3),
            move=candidate["move"],
            fen=candidate["fen"],
            level=level,
        )

        self.tree[child_name] = node

        return node

    def get_best_move(self):
        parent = None
        node: Node

        # walk over all leafs reversed (from furthest children) and backprop scores on parents
        ordered_nodes = [node for node in LevelOrderIter(self.root)]
        for node in reversed(ordered_nodes):
            # if node.level < self.depth:
            #     try:
            #         CommonDataManager.remove_node_memory(node.name)
            #     except FileNotFoundError:
            #         pass

            if node.level < 1:
                break

            if node.parent != parent:
                parent = node.parent
                parent.score = node.score
            else:
                is_white_move = (node.level % 2 == 0 and self.turn) or (node.level % 2 == 1 and not self.turn)
                if is_white_move:
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
