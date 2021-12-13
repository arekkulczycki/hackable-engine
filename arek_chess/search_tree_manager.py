# -*- coding: utf-8 -*-
"""
Module_docstring.
"""
import ctypes
import math
import time
import traceback
from functools import reduce
from statistics import mean

from anytree import Node, RenderTree, LevelOrderIter
from chess import Move

from arek_chess.board.board import Board
from arek_chess.dispatcher import Dispatcher
from arek_chess.messaging import Queue


class PrintableNode(Node):
    def __repr__(self):
        return f"Node({self.level}, {self.move}, {self.score})"


class SearchTreeManager:
    """
    Class_docstring
    """

    SLEEP = 0.001
    ROOT_NAME = "0"

    def __init__(self, fen: str, turn: bool, depth: int = 6):
        self.turn = turn
        self.depth = depth

        self.root = PrintableNode(self.ROOT_NAME, turn=turn, score=0, fen=fen, level=0, move=None)
        self.tree = {self.ROOT_NAME: self.root}

        self.node_queue = Queue("node")
        self.candidates_queue = Queue("candidate")

        self.board = Board()

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
            # self.board._set_board_fen(parent.fen.split(" ")[0])
            parent_name_split = parent.name.split(".")
            level = len(parent_name_split)
            turn_after = level % 2 == 0 if self.turn else level % 2 == 1
            # self.board.turn = not turn_after

            for i, candidate in enumerate(candidates):
                node, is_dominated = self.create_node(i, candidate, parent, level, turn_after)
                if is_dominated:
                    dominated += 1
                    continue

                if node.level < self.depth:
                    sent += 1
                    self.node_queue.put((node.name, node.fen, turn_after))

            returned += 1

        # print(RenderTree(self.root))

        print(sent, returned, dominated)

        print(self.get_best_move())

    def get_node(self, node_name):
        return self.tree[node_name]

    def create_node(self, i, candidate, parent, level, turn_after):
        parent_name = parent.name
        score = candidate["score"]

        move = candidate["move"]
        fen = candidate["fen"]
        child_name = f"{parent_name}.{i}"

        # self.board.push(Move.from_uci(move))
        # fen = self.board.fen().split(" ")[0]
        # self.board.pop()
        #
        # if fen_ != fen:
        #     print(fen, fen_)

        node = PrintableNode(
            child_name,
            parent=parent,
            score=round(score, 3),
            move=move,
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
            if trend[0] < trend[1] < trend[2] and trend[1] < 0:
                return True
        else:
            if trend[0] > trend[1] > trend[2] and trend[1] > 0:
                return True

        if color:
            if all([delta < 0 for delta in trend]) and trend[2] / trend[1] > trend[1] / trend[0]:
                return True
            elif trend[0] < trend[1] < trend[2] and trend[1] < 0:
                return True
        else:
            if all([delta > 0 for delta in trend]) and trend[2] / trend[1] > trend[1] / trend[0]:
                return True
            elif trend[0] > trend[1] > trend[2] and trend[1] > 0:
                return True

        return False

    def get_trend(self, score, parent: Node):
        """recent go first"""
        consecutive_scores = [score, *self.get_consecutive_scores(parent)]
        averages = [(consecutive_scores[i] + consecutive_scores[i+1] / 2) for i in range(4)]
        deltas = [averages[i + 1] - averages[i] for i in range(3)]
        return deltas

    def get_consecutive_scores(self, node: Node):
        # TODO: WTF, just go like parent.parent.parent ???
        parents = node.name.split('.')
        last_name = ".".join(parents[:-4]) or "0"  # TODO: don't know why it required fallback value of 0
        third_name = f"{last_name}.{parents[-4]}"
        second_name = f"{third_name}.{parents[-3]}"
        first_name = f"{second_name}.{parents[-2]}"
        return [self.get_node(name).score for name in [first_name, second_name, third_name, last_name]]

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
