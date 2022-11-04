# -*- coding: utf-8 -*-
"""
Test board calculations.
"""
from time import perf_counter
from unittest import TestCase

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.optimized_eval import OptimizedEval


class BoardTest(TestCase):
    """
    Test board calculations.
    """

    def test_threats(self):
        """"""

        fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
        board = Board(fen)

        D2 = 11  # file attack by rook
        G4 = 30  # diagonal attack by queen
        attackers = board.threats_mask(True, board.mask_to_square(board.bishops))

        assert set(board.mask_to_squares(attackers)) == {D2, G4}

    def test_get_pawn_value(self):
        """"""

        assert Board.get_pawn_value(7, True) > 8.9

        assert Board.get_pawn_value(0, False) > 8.9

        assert Board.get_pawn_value(0, True) == 1
        assert Board.get_pawn_value(1, True) == 1
        assert Board.get_pawn_value(2, True) == 1

        assert Board.get_pawn_value(7, False) == 1
        assert Board.get_pawn_value(6, False) == 1
        assert Board.get_pawn_value(5, False) == 1

    def test_get_mobility(self):
        fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
        board = Board(fen)

        assert board.get_mobility(True) == 9 + 15

        assert board.get_mobility(False) == 7

    def test_get_threats(self):
        fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
        board = Board(fen)

        assert board.get_threats(True) == 6

        assert board.get_threats(False) == 9

        fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
        board = Board(fen)

        assert board.get_threats(True) == 28

        assert board.get_threats(False) == 25

    def test_colored_squares(self):
        fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
        board = Board(fen)

        assert board.pieces_on_light_squares(True) == 4
        assert board.pieces_on_light_squares(False) == 3
        assert board.pawns_on_light_squares(True) == 2
        assert board.pawns_on_light_squares(False) == 5

        assert board.pieces_on_dark_squares(True) == 3
        assert board.pieces_on_dark_squares(False) == 4
        assert board.pawns_on_dark_squares(True) == 5
        assert board.pawns_on_dark_squares(False) == 2

    def test_direct_threats(self):
        fen = "8/8/2b5/5q2/3N4/1p6/R1P5/8 w - - 0 1"
        board = Board(fen)

        assert board.get_direct_threats(True) == 12
        assert board.get_direct_threats(False) == 5

    def test_king_mobility(self):
        fen = "k1b5/Bpp5/8/8/8/7P/5Bp1/6K1 w - - 0 1"
        board = Board(fen)

        assert board.get_king_mobility(True) == 3
        assert board.get_king_mobility(False) == 2

    def test_perf(self):
        fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
        board = Board(fen)
        eval = OptimizedEval()

        k = 0
        t0 = perf_counter()
        while perf_counter() - t0 < 1:
            eval.get_score(board, "", 0)
            k += 1

        print(k)
        assert k > 15000
