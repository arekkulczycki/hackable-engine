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
        (None, "a2a3", True, -1),
        (None, "a2a4", True, 1),
        (None, "b1c3", True, 2),
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
        board = Board(fen) if fen else Board()
        board.turn = color
        captured_piece_type = board.piece_type_at(move.to_square) or 0

        initial_board = board.copy()

        delta = board.get_mobility_delta(move, captured_piece_type)

        assert delta == result

        # test no side effects
        assert initial_board == board

    @parameterized.expand([
        (None, "a2a3", True),
        (None, "a2a4", True),
        (None, "b1c3", True),
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR", "e7e5", False),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR", "d2d4", True),
        ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR", "d7d5", False),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "g4f5", True),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "f4g5", True),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "f5g4", False),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "g5f4", False),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "c5d6", True),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", "b6d5", True),
        ("rnbqkb2/1p2p2p/1Npp1r1n/2P2pp1/1PQ1BPP1/1p1P1N2/P3P2P/R1B1K2R", "b3a2", False),
        ("rnbqkb2/1p2p2p/1Npp1r1n/2P2pp1/1PQ1BPP1/1p1P1N2/P3P2P/R1B1K2R", "h6g4", False),
    ])
    def test_get_pawn_mobility_delta(self, fen, move, color):
        move = Move.from_uci(move)
        board = Board(fen) if fen else Board()
        board.turn = color

        initial_board = board.copy()

        pawn_mobility_before = board.get_pawn_mobility()
        board.push(move)
        pawn_mobility_after = board.get_pawn_mobility()
        actual_delta = pawn_mobility_after - pawn_mobility_before
        board.pop()

        fast_delta = board.get_pawn_mobility_delta(move)
        # print(actual_delta, fast_delta)
        assert actual_delta == fast_delta

        # test no side effects
        assert initial_board == board
