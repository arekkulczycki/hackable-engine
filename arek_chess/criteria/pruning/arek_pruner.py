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
        if parent_node.level < 3:
            return False

        # prune based on grandparent which already has the score backpropped
        grandparent = parent_node.parent
        stats = tree_stats.get(grandparent.level)
        grandmedian = stats.get("median")
        grandmean = stats.get("mean")
        grandstdev = stats.get("stdev")
        if color and score < grandmedian:
            # print(f"pruning, color: {color}, move: {grandparent.move}, score: {grandparent.score}, median: {median}")
            return True
        elif not color and score > grandmedian:
            # print(f"pruning, color: {color}, move: {grandparent.move}, score: {grandparent.score}, median: {median}")
            return True

        # if parent_node.level >= depth:
        #     if self.is_capture_fest_harmful(score, color, captured, parent_node):
        #         return True  # TODO: find smarter rule when engine is ready for it

        # stats are available for parent only when children already analysed
        #   so now I can discard node based on discarding the entire parent node
        # if self.parent_worse_than_prev_generation(tree_stats, parent_node, color):
        #     return True

        # if self.is_worse_than_prev_generations(
        #     tree_stats, score, parent_node, color
        # ):
        #     return True
        #
        # if self.is_not_promising(score, parent_node, color):
        #     return True

        return False

    def parent_worse_than_prev_generation(self, tree_stats, parent: Node, color: bool):
        last_comparative_score = tree_stats.get(parent.level - 2).get("median")

        # if color is WHITE then I compare BLACK score to it's grandparent level median
        return parent.score > last_comparative_score if color else parent.score < last_comparative_score

    @staticmethod
    def is_worse_than_prev_generations(tree_stats, score, parent: Node, color: bool):
        stats = tree_stats.get(parent.level - 1)
        median = stats.get("median")

        # grandstats = tree_stats.get(parent.level - 3)
        # grandmedian = grandstats.get("median")
        grandmedian = score

        return (
            (score < median or score < grandmedian)
            if color
            else (score > median or score > grandmedian)
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

    # @staticmethod
    # def is_capture_fest_harmful(score: float, color: bool, captured: int, parent_node: Node):
    #     opp_cp = parent_node.captured
    #     opp_captured_first = parent_node.parent.parent.captured
    #
    #     if (
    #         opp_captured_first
    #         and captured < opp_cp
    #         and parent_node.parent.captured < opp_cp
    #     ):
    #         if (
    #             score < parent_node.parent.score
    #             if color
    #             else score > parent_node.parent.score
    #         ):
    #             return True
    #
    #     if (
    #         opp_captured_first > captured
    #         and opp_captured_first > parent_node.parent.captured
    #         and opp_cp > captured
    #         and opp_cp > parent_node.parent.captured
    #     ):
    #         if (
    #             score < parent_node.parent.score
    #             if color
    #             else score > parent_node.parent.score
    #         ):
    #             return True
    #
    #     return False
