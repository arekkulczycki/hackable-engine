# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM
from typing import Dict, List

import kmeans1d
from numpy import absolute as np_absolute, mean as np_mean, std as np_std
from numpy.random import choice
from pyinstrument import Profiler

from arek_chess.main.controller import DEPTH
from arek_chess.board.board import Move
from arek_chess.utils.memory_manager import (
    MemoryManager,
    remove_shm_from_resource_tracker,
)
from arek_chess.utils.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker

CLUSTER_2_3_THRESHOLD = 9
CLUSTER_3_4_THRESHOLD = 15
CLUSTER_4_5_THRESHOLD = 25


class SelectionWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.0005

    def __init__(
        self, selector_queue: Queue, candidates_queue: Queue
    ):
        super().__init__()

        self.selector_queue = selector_queue
        self.candidates_queue = candidates_queue

        self.groups = {}

    def run(self):
        """

        :return:
        """

        remove_shm_from_resource_tracker()

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.hanging_board_memory = set()

        # self.pickler = Pickler(with_refs=False, protocol=5)

        while True:
            scored_move_item = self.selector_queue.get()

            if scored_move_item:
                (
                    node_name,
                    size,
                    move,
                    turn_after,
                    captured_piece_type,
                    score,
                ) = scored_move_item
                if node_name not in self.groups:
                    self.groups[node_name] = {"size": size, "moves": [], "captures": []}
                    self.hanging_board_memory.add(node_name)

                moves = self.groups[node_name]["moves"]
                captures = self.groups[node_name]["captures"]
                candidate = {
                    "move": move,
                    "score": score,
                    "captured": captured_piece_type,
                }
                if not captured_piece_type:
                    moves.append(candidate)
                else:
                    captures.append(candidate)

                if len(moves) + len(captures) == size:
                    candidates = self.select(moves, turn_after) + self.select(
                        captures, turn_after
                    )
                    board = MemoryManager.get_node_board(node_name)
                    self.candidates_queue.put(
                        (node_name, self.get_candidates(node_name, not turn_after, candidates, board))
                    )

                    del self.groups[node_name]
                    # erase memory for params and board for the parent node of all candidates
                    MemoryManager.remove_node_memory(node_name)
                    try:
                        self.hanging_board_memory.remove(node_name)
                    except KeyError:
                        continue

            else:
                # nothing done, wait a little
                time.sleep(self.SLEEP)

    def get_candidates(self, node_name: str, turn_before: bool, candidates: List[Dict], board) -> List[Dict]:
        # board = MemoryManager.get_node_board(node_name)
        board.turn = turn_before

        for i, candidate in enumerate(candidates):
            # board.push(Move.from_uci(candidate["move"]))
            state = board.push_no_stack(Move.from_uci(candidate["move"]))

            candidate["i"] = i
            candidate_name = f"{node_name}.{i}"
            MemoryManager.set_node_board(candidate_name, board)

            self.hanging_board_memory.add(candidate_name)

            # board.pop()
            board.light_pop(state)
        return candidates

    def select(self, moves: List[Dict], turn: bool) -> List[Dict]:
        """

        :return:
        """

        lcan = len(moves)

        if lcan > 2:
            return self.select_best_group(moves, not turn, lcan)
        elif lcan:
            return moves
        else:
            return []

    def select_best_group(
        self, candidates: List[Dict], turn: bool, lcan: int, repeated=False
    ) -> List[Dict]:
        """
        Take all scored moves and select the strongest subset.

        TODO: messy legacy algorithm, rethink from scratch

        :param candidates: candidate moves
        :param turn: true for white's turn, false for black's
        :param lcan: number of candidate moves
        :param repeated:
        :return:
        """
        # self.selects += 1

        outlier_candidates = (
            []
        )  # self.find_outliers(candidates, lcan, turn) if not repeated else []

        k = (
            2
            if lcan < CLUSTER_2_3_THRESHOLD
            else 3
            if lcan < CLUSTER_3_4_THRESHOLD
            else 4
            if lcan < CLUSTER_4_5_THRESHOLD
            else 5
        )  # number of clusters
        clusters, centroids = kmeans1d.cluster(
            [cand["score"] for cand in candidates], k
        )

        for step in range(
            k
        ):  # sometimes candidates are so similar that they are not split into k centroids
            first_candidates = [
                candidates[i]
                for i, c in enumerate(clusters)
                if (turn and c == k - (step + 1)) or (not turn and c == step)
            ]
            if first_candidates:
                break

        if k > 2 and len(first_candidates + outlier_candidates) < 3:
            second_candidates = [
                candidates[i]
                for i, c in enumerate(clusters)
                if (turn and c == k - (step + 2)) or (not turn and c == step + 1)
            ]
            best_candidates = first_candidates + second_candidates

            lcan = len(best_candidates)
            if lcan >= CLUSTER_2_3_THRESHOLD and not repeated:
                return (
                    self.select_best_group(
                        second_candidates.copy(), turn, lcan, repeated=True
                    )
                    + first_candidates
                    + outlier_candidates
                )
            elif lcan >= CLUSTER_2_3_THRESHOLD:
                # TODO: wtf random?
                return (
                    [choice(second_candidates) for _ in range(2)]
                    + first_candidates
                    + outlier_candidates
                )

            return best_candidates + outlier_candidates

        lcan = len(first_candidates)
        if lcan >= CLUSTER_2_3_THRESHOLD and not repeated:
            return (
                self.select_best_group(
                    first_candidates.copy(), turn, lcan, repeated=True
                )
                + outlier_candidates
            )
        elif lcan >= CLUSTER_2_3_THRESHOLD:
            # TODO: wtf random?
            return [choice(first_candidates) for _ in range(3)] + outlier_candidates

        return first_candidates + outlier_candidates

    @staticmethod
    def find_outliers(
        candidates: List[Dict], lcan: int, turn: bool, threshold=3
    ):  # deliberately mutable arg!
        outlier_candidates = []
        scores = [cand["score"] for cand in candidates]
        # TODO: optimize further? maybe both below should be calculated at once
        mean = np_mean(scores)
        stdev = np_std(scores)

        for i in reversed(range(lcan)):
            score = candidates[i]["score"]
            if np_absolute(score - mean) > threshold * stdev:
                if (turn and score > mean) or (
                    not turn and score < mean
                ):  # it's either high-score or low-score outlier
                    outlier_candidates.append(candidates[i])
                del candidates[i]
        return outlier_candidates

    def before_exit(self, *args):
        """"""

        print("cleaning...")

        for node_name in self.hanging_board_memory:
            try:
                MemoryManager.remove_node_board_memory(node_name)
            except:
                print(f"error erasing: {node_name}")
                continue

        print("cleaning done")

        if getattr(self, "should_profile_code", False):
            self.profiler.stop()
            self.profiler.print(show_all=True)

        exit(0)

    def profile_code(self):
        self.profiler = Profiler()
        self.profiler.start()

        self.should_profile_code = True
