# -*- coding: utf-8 -*-
from itertools import cycle
from time import perf_counter
from unittest import TestCase

from hackable_engine.board.hex.hex_board import HexBoard as Board
from hackable_engine.criteria.evaluation.hex.simple_eval import SimpleEval


class TestSimpleEval(TestCase):
    def test_get_score_speed(self) -> None:
        boards = cycle(
            [
                Board("e5c11i5m13a13"),
                Board("d5c7b11f8k5j6h4f4j11"),
                Board("h8l10i5f4c4b8c11f11f7k7"),
                Board("f12d8f5h8g9j4k2b4c13c10c7h6m6k10j8i11f3e2"),
            ]
        )

        eval = SimpleEval()
        t0 = perf_counter()
        for i in range(10000):
            score = eval.get_score(next(boards), False, None)
        t = perf_counter() - t0
        print(t, f"{10000 / t} per second")

    def get_score(self, notation):
        board = Board(notation)
        return SimpleEval().get_score(board, False, None)


# print(TestSimpleEval().get_score("e5c11i5m13a13"))
TestSimpleEval().test_get_score_speed()
