# -*- coding: utf-8 -*-
"""
Tests for the board module.
"""
from random import randint
from typing import List
from unittest import TestCase

from parameterized import parameterized

from arek_chess.main.search_tree_manager import (
    BackpropNode,
    SearchTreeManager,
    GetBestMoveMixin,
)


def _create_node(name: str, score: float = None):
    """"""

    if score is None:
        score = randint(-1000, 1000)

    return BackpropNode(name, color=True, level=0, score=score)


class BoardTest(TestCase):
    """
    Tests for the board module.
    """

    @staticmethod
    def _create_node(name: str, score: float = None):
        """"""

        if score is None:
            score = randint(-1000, 1000)

        return BackpropNode(name, color=True, level=0, score=score)

    def _create_node_tree(self, node_list):
        """"""

    # @parameterized.expand([
    #     ["", "", "", "", "", "", "", ""],
    #     [],
    #     [],
    # ])
    # def test_get_best_move(self, root: BackpropNode):
    #     """"""

    @parameterized.expand(
        [
            [
                [
                    _create_node("1", 0.13),
                    _create_node("2", -0.13),
                    _create_node("3", 123.97),
                    _create_node("4", 123),
                    _create_node("5", -1000),
                ],
                123.97,
                True,
            ],
            [
                [
                    _create_node("1", 0.13),
                    _create_node("2", -0.13),
                    _create_node("3", 123.97),
                    _create_node("4", 123),
                    _create_node("5", -1000),
                ],
                -1000,
                False,
            ],
            [
                [
                    _create_node("1", 0.13),
                    _create_node("2", -0.13),
                    _create_node("3", 12.1234559),
                    _create_node("4", 123),
                    _create_node("5", 12.1234567),
                ],
                12.1234567,
                True,
            ],
        ]
    )
    def test_get_the_best_1st_level(
        self, children: List[BackpropNode], best_score: float, starting_color: bool
    ) -> None:
        root = self._create_node("0", 0)
        for child in children:
            child.parent = root

        assert (
            GetBestMoveMixin._get_the_best_1st_level(root, starting_color)[0]
            == best_score
        )
