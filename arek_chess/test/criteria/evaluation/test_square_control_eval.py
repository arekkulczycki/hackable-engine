# -*- coding: utf-8 -*-
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

    def get_score(self, fen):
        board = Board(fen)
        return SquareControlEval().get_score(board, board.is_check(), None)


# print(TestSquareControlEval().get_score("2b2rk1/1p2q2p/2pbPpp1/p7/2P1B2Q/P7/1B3PPP/4R1K1 w - - 0 22"))
print(TestSquareControlEval().get_score("2b2rk1/1p2q2p/3bPpp1/p2P4/7Q/P7/1B3PPP/4R1K1 b - - 0 23"))
