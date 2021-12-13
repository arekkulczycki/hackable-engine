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

    @parameterized.expand([
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "e2e3", True, 10),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "d4d5", True, -1),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "g5f6", True, 6),
        ("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR", "c8d7", False, 1),
        ("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR", "b8d7", False, 6),
        ("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR", "c8h3", False, -4),
        ("rnbqkbnr/ppp2ppp/8/3pp2Q/3PP3/8/PPP2PPP/RNB1KBNR", "d8g5", False, -12),
    ])
    def test_mobility_delta(self, fen, move, color, result):
        move = Move.from_uci(move)
        board = Board(fen)
        board.turn = color
        captured_piece = board.piece_at(move.to_square)
        captured_piece_type = captured_piece.piece_type if captured_piece else 0

        delta = board.get_mobility_delta(move, captured_piece_type)
        print(delta)
        assert delta == result
