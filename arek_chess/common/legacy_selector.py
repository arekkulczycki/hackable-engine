from typing import Dict, Union, List

import kmeans1d
from numpy import absolute as np_absolute, mean as np_mean, std as np_std
from numpy.random import choice

CLUSTER_2_3_THRESHOLD = 9
CLUSTER_3_4_THRESHOLD = 15
CLUSTER_4_5_THRESHOLD = 25


class LegacySelector:
    """"""

    def select(
        self, candidates: List[Dict[str, Union[str, int, float]]], color: bool
    ) -> List[Dict[str, Union[str, int, float]]]:
        captures = []
        non_captures = []

        for move in candidates:
            if move.get("captured", 0):
                captures.append(move)
            else:
                non_captures.append(move)

        return self._select(non_captures, color) + self._select(captures, color)

    def _select(
        self, moves: List[Dict[str, Union[str, int, float]]], color: bool
    ) -> List[Dict[str, Union[str, int, float]]]:
        lcan = len(moves)

        if lcan > 2:
            return self.select_best_group(moves, color, lcan)
        elif lcan:
            return moves
        else:
            return []

    def select_best_group(
        self, candidates: List[Dict], turn: bool, lcan: int, repeated=False
    ) -> List[Dict]:
        """
        Take all scored moves and select the strongest subset.

        :param candidates: candidate moves
        :param turn: true for white's turn, false for black's
        :param lcan: number of candidate moves
        """

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
