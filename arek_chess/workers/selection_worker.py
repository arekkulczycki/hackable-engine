# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM
from typing import Dict, List

import kmeans1d
from chess import Move
from numpy import absolute as np_absolute, mean as np_mean, std as np_std
from numpy.random import choice
from pyinstrument import Profiler

from arek_chess.utils.common_data_manager import CommonDataManager
from arek_chess import DEPTH
from arek_chess.utils.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker

CLUSTER_2_3_THRESHOLD = 9
CLUSTER_3_4_THRESHOLD = 15
CLUSTER_4_5_THRESHOLD = 25


class SelectionWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.001

    def __init__(
        self, selector_queue: Queue, candidates_queue: Queue, to_erase_queue: Queue
    ):
        super().__init__()

        self.selector_queue = selector_queue
        self.candidates_queue = candidates_queue
        self.to_erase_queue = to_erase_queue

        self.groups = {}

    def run(self):
        """

        :return:
        """

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.hanging_nodes = set()

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

                moves = self.groups[node_name]["moves"]
                captures = self.groups[node_name]["captures"]
                if not captured_piece_type:
                    moves.append(
                        {
                            "node_name": node_name,
                            "move": move,
                            "turn": turn_after,
                            "score": score,
                            "is_capture": False,
                        }
                    )
                else:
                    captures.append(
                        {
                            "node_name": node_name,
                            "move": move,
                            "turn": turn_after,
                            "score": score,
                            "is_capture": True,
                        }
                    )

                if len(moves) + len(captures) == size:
                    candidates = self.select(moves, turn_after) + self.select(
                        captures, turn_after
                    )
                    self.candidates_queue.put(
                        self.assign_fen(node_name, not turn_after, candidates)
                    )

                    del self.groups[node_name]
                    CommonDataManager.remove_node_memory(node_name)

            else:
                node_name_to_erase = self.to_erase_queue.get()
                if node_name_to_erase:
                    CommonDataManager.remove_node_board_memory(node_name_to_erase)
                    try:
                        self.hanging_nodes.remove(node_name_to_erase)
                    except KeyError:
                        continue
                else:
                    # nothing done, wait a little
                    time.sleep(self.SLEEP)

    def assign_fen(self, node_name, turn_before, candidates):
        board = CommonDataManager.get_node_board(node_name)
        board.turn = turn_before

        for i, candidate in enumerate(candidates):
            board.push(Move.from_uci(candidate["move"]))

            candidate["i"] = i
            # TODO: OH SHIT it loses information about castling rights
            candidate_name = f"{node_name}.{i}"
            CommonDataManager.set_node_board(candidate_name, board)
            if len(node_name.split(".")) >= DEPTH:
                self.hanging_nodes.add(candidate_name)

            board.pop()
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

        for node_name in self.hanging_nodes:
            try:
                CommonDataManager.remove_node_board_memory(node_name)
            except:
                continue

        exit(0)

    def profile_code(self):
        profiler = Profiler()
        profiler.start()

        def before_exit(*args):
            """"""
            for node_name in self.hanging_nodes:
                try:
                    CommonDataManager.remove_node_board_memory(node_name)
                except:
                    continue

            profiler.stop()
            profiler.print(show_all=True)
            exit(0)

        signal(SIGTERM, before_exit)
