# -*- coding: utf-8 -*-
"""
Tests for the board module.
"""

from time import perf_counter
from unittest import TestCase

from larch.pickle.pickle import dumps, loads
from parameterized import parameterized

from hackable_engine.board.legacy_board.board import Board, Move


class BoardTest(TestCase):
    """
    Tests for the board module.
    """

    @parameterized.expand([
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R",),
        ("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR",),
    ])
    def test_pickling(self, fen: str):
        board = Board(fen)
        board_bytes = dumps(board)
        recovered_board = loads(board_bytes)

        assert board == recovered_board

    @parameterized.expand([
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R",),
        ("rnbqkbnr/ppp2ppp/8/3pp3/3PP3/2N5/PPP2PPP/R1BQKBNR",),
    ])
    def test_outcome(self, fen: str):
        board = Board(fen)

        t_0 = perf_counter()
        outcomes = [board.simple_outcome() for _ in range(10000)]
        print(perf_counter() - t_0)

    @parameterized.expand([
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "h2h4", 3.907, 0.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "g5f6", -3.0, -10.0),
        ("r2qk2r/pppb1ppp/2n1pn2/6B1/1bBP4/2N1PN2/PP3PPP/R2QK2R", "d1a4", -12, 0.0),
    ])
    def test_get_safety_delta(self, fen, move, result_white, result_black):
        move = Move.from_uci(move)
        board = Board(fen)
        piece_type = board.piece_at(move.from_square).piece_type
        captured_piece = board.piece_at(move.to_square)
        captured_piece_type = captured_piece.piece_type if captured_piece else 0

        white = board.get_safety_delta(True, move, piece_type, captured_piece_type)
        black = board.get_safety_delta(False, move, piece_type, captured_piece_type)

        assert round(white, 3) == result_white
        assert round(black, 3) == result_black

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

        assert round(white, 3) == result_white
        assert round(black, 3) == result_black

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

        captured_piece_type = board.get_captured_piece_type(move)
        fast_delta = board.get_pawn_mobility_delta(move, captured_piece_type)

        assert actual_delta == fast_delta

        # test no side effects
        assert initial_board == board

    # @parameterized.expand([
    #     ("r1b1k1nr/ppppqppp/2n1p3/8/3P4/3Q4/PPP1PPPP/RN2KBNR w KQkq - 0 5", 35.047, 25.047, 0, 35.005, 33.013, 0)
    # ])
    # def test_get_material_and_safety(self, fen, result_white_material, result_white_safety, result_white_under_attack,
    #                                  result_black_material, result_black_safety, result_black_under_attack):
    #     board = Board(fen)
    #
    #     white_material, white_safety, white_under_attack = board.get_material_and_safety(True)
    #     black_material, black_safety, black_under_attack = board.get_material_and_safety(False)
    #
    #     assert round(white_material, 3) == result_white_material
    #     assert round(white_safety, 3) == result_white_safety
    #     assert round(white_under_attack, 3) == result_white_under_attack
    #     assert round(black_material, 3) == result_black_material
    #     assert round(black_safety, 3) == result_black_safety
    #     assert round(black_under_attack, 3) == result_black_under_attack

    @parameterized.expand([
        ("rnbqkbnr/pppp1ppp/8/4p3/3PP3/8/PPP2PPP/RNBQKBNR", 12, 10),
        ("rnbqkb2/1p2p2p/1Npp1r1n/p1P2pp1/1PQ1BPP1/3P1N2/P3P2P/R1B1K2R", 18, 16),
        ("rnbqkbnr/ppp1pp1p/8/3P4/3P4/4p3/PP4PP/RNBQKBNR", 25, 18),
    ])
    def test_get_space(self, fen, result_white_space, result_black_space):
        board = Board(fen)

        assert board.get_space(True) == result_white_space
        assert board.get_space(False) == result_black_space

        # t0 = perf_counter()
        # for i in range(1000000):
        #     board.get_space(True)
        #     board.get_space(False)
        # print(f"speed: {perf_counter() - t0}")

    # @parameterized.expand(["rnbqkbnr/ppp1pp1p/8/3P4/3P4/4p3/PP4PP/RNBQKBNR"])
    # def test_get_material_simple(self, fen):
    #     board = Board(fen)
    #
    #     t0 = perf_counter()
    #     for i in range(1000000):
    #         board.get_material_simple(True)
    #         board.get_material_simple(False)
    #     print(f"speed: {perf_counter() - t0}")

# unittest.main()
