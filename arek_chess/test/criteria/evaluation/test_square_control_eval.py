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
