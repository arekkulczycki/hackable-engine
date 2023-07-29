# -*- coding: utf-8 -*-
from itertools import cycle
from time import perf_counter
from unittest import TestCase

from parameterized import parameterized

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.square_control_eval import SquareControlEval


class TestSquareControlEval(TestCase):

    @parameterized.expand([
        ["rnbqkbnr/pp1ppppp/2p5/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2", 9.0],  # white 27, black 18
    ])
    def test_get_empty_square_control(self, fen, result) -> None:
        board = Board(fen)
        square_control_diff = board.get_square_control_map_for_both()
        assert SquareControlEval._get_empty_square_control(board, square_control_diff) == result

    def test_get_score_speed(self) -> None:
        boards = cycle(
            [
                Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
                Board("r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"),
                Board("rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"),
                Board("r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9"),
                Board("rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8"),
                Board("2b2rk1/1p2q2p/2pbPpp1/p7/2P1B2Q/P7/1B3PPP/4R1K1 w - - 0 22"),
                Board("2b2rk1/1p2q2p/3bPpp1/p2P4/7Q/P7/1B3PPP/4R1K1 b - - 0 23"),
            ]
        )

        eval = SquareControlEval()
        t0 = perf_counter()
        for i in range(10000):
            score = eval.get_score(next(boards), False, None)
        t = perf_counter() - t0
        print(t, f"{10000 / t} per second")

    def get_score(self, fen):
        board = Board(fen)
        return SquareControlEval().get_score(board, board.is_check(), None)


# print(TestSquareControlEval().get_score("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
TestSquareControlEval().test_get_score_speed()
