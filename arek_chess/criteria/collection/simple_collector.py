# -*- coding: utf-8 -*-
"""
Simple prioritizer that returns directly all the unchanged candidates.
"""

from typing import Dict, Union, List

from arek_chess.criteria.collection.base_collector import BaseCollector


class SimpleCollector(BaseCollector):
    """
    Prioritize high value caputures
    """

    def order(
        self, candidates: List[Dict[str, Union[str, int, float]]], color: bool
    ) -> List[Dict[str, Union[str, int, float]]]:
        """

        :param candidates: list of all legal moves in the position
        :param color: color of the player to choose from above candidates

        :return: unchanged list of candidates
        """

        return sorted(candidates, key=lambda cand: cand["is_check"], reverse=True)
