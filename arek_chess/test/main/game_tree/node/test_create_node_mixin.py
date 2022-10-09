# -*- coding: utf-8 -*-
"""
Tests for CreateNodeMixin class.
"""

import random
from time import perf_counter
from unittest import TestCase

from parameterized import parameterized

from arek_chess.main.game_tree.node.node import Node
from arek_chess.main.game_tree.node.create_node_mixin import CreateNodeMixin


class TestCreateNodeMixin(TestCase):
    """"""

    @parameterized.expand([
        (True,),
        (False,),
    ])
    def test_get_space(self, color: bool):
        mixin = CreateNodeMixin()

        nodes = [Node("", random.random(), 0, color) for _ in range(25)]

        t0 = perf_counter()
        for i in range(100000):
            mixin.get_best_node(nodes, True)
        print(f"time python: {perf_counter() - t0}")

        t0 = perf_counter()
        for i in range(100000):
            mixin.select_promising_node(nodes, True)
        print(f"time select: {perf_counter() - t0}")
