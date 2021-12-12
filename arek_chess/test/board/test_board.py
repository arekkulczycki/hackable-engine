# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from unittest import TestCase

from parameterized import parameterized
from chess import Move

from arek_chess.board.board import Board


class BoardTest(TestCase):
    """
    Class_docstring
    """

    @parameterized.expand([
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "h2h4", 4.0, 0.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "g5f6", -3.0, -10.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "d1a4", -25.5, 0.0),
    ])
    def test_get_safety_delta(self, fen, move, result_white, result_black):
        move = Move.from_uci(move)
        board = Board(fen)
        piece_type = board.piece_at(move.from_square).piece_type
        captured_piece = board.piece_at(move.to_square)
        captured_piece_type = captured_piece.piece_type if captured_piece else 0

        white = board.get_safety_delta(True, move, piece_type, captured_piece_type)
        black = board.get_safety_delta(False, move, piece_type, captured_piece_type)

        assert white == result_white
        assert black == result_black

    @parameterized.expand([
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "e2e3", 0.0, 0.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "d4d5", 1.0, 3.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "g5f6", 6.0, 7.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "d1a4", 0.0, 7.0),
    ])
    def test_get_under_attack_delta(self, fen, move, result_white, result_black):
        move = Move.from_uci(move)
        board = Board(fen)
        piece_type = board.piece_at(move.from_square).piece_type
        captured_piece = board.piece_at(move.to_square)
        captured_piece_type = captured_piece.piece_type if captured_piece else 0

        white = board.get_under_attack_delta(True, move, piece_type, captured_piece_type)
        black = board.get_under_attack_delta(False, move, piece_type, captured_piece_type)

        assert white == result_white
        assert black == result_black
