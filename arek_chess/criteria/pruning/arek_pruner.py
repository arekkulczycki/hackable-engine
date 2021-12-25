# -*- coding: utf-8 -*-

from typing import Dict

from anytree import Node

from arek_chess.criteria.pruning.base_pruner import BasePruner


class ArekPruner(BasePruner):
    """"""

    def should_prune(
        self,
        tree_stats: Dict[str, Dict[str, float]],
        score: float,
        parent_node: Node,
        color: bool,
        captured: int,
        depth: int,
    ) -> bool:
        if not (parent_node.level >= 3 and parent_node.level < depth - 1):
            return False

        if parent_node.level >= depth:
            if self.is_good_enough_capture(score, parent_node, color):
                return True

        not_promising = self.is_not_promising(score, parent_node, color)

        is_worse_than_last_generation = self.is_worse_than_last_generation(
            tree_stats, score, parent_node, color
        )

        return is_worse_than_last_generation or not_promising

    @staticmethod
    def is_good_enough_capture(score: float, parent_node: Node, color: bool):
        """"""

        if parent_node.captured and parent_node.parent.captured and parent_node.parent.parent.captured and not parent_node.parent.parent.parent.captured:
            if score > parent_node.parent.score if color else score < parent_node.parent.score:
                return True

        return False

    @staticmethod
    def is_worse_than_last_generation(tree_stats, score, parent: Node, color: bool):
        stats = tree_stats.get(parent.level - 1)
        median = stats.get("median")

        return score < median if color else score > median

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
