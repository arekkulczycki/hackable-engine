# -*- coding: utf-8 -*-
from time import perf_counter
from unittest import TestCase

from parameterized import parameterized

from hackable_engine.board.chess.chess_board import ChessBoard as Board
from hackable_engine.criteria.evaluation.chess.square_control_eval import SquareControlEval


class BoardTest(TestCase):
    """
    Test board calculations.
    """

    # def test_threats(self):
    #     """"""
    #
    #     fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
    #     board = Board(fen)
    #
    #     D2 = 11  # file attack by rook
    #     G4 = 30  # diagonal attack by queen
    #     attackers = board.threats_mask(True, board.mask_to_square(board.bishops))
    #
    #     assert set(board.mask_to_squares(attackers)) == {D2, G4}
    #
    # def test_get_pawn_value(self):
    #     """"""
    #
    #     assert Board.get_pawn_value(7, True) > 8.9
    #
    #     assert Board.get_pawn_value(0, False) > 8.9
    #
    #     assert Board.get_pawn_value(0, True) == 1
    #     assert Board.get_pawn_value(1, True) == 1
    #     assert Board.get_pawn_value(2, True) == 1
    #
    #     assert Board.get_pawn_value(7, False) == 1
    #     assert Board.get_pawn_value(6, False) == 1
    #     assert Board.get_pawn_value(5, False) == 1
    #
    # def test_get_mobility(self):
    #     fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
    #     board = Board(fen)
    #
    #     assert board.get_mobility(True) == 9 + 15
    #
    #     assert board.get_mobility(False) == 7
    #
    # def test_get_threats(self):
    #     fen = "8/3b4/8/5P2/3P2Q1/8/3R4/8 w - - 0 1"
    #     board = Board(fen)
    #
    #     assert board.get_threats(True) == 6
    #
    #     assert board.get_threats(False) == 9
    #
    #     fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
    #     board = Board(fen)
    #
    #     assert board.get_threats(True) == 28
    #
    #     assert board.get_threats(False) == 25
    #
    # def test_colored_squares(self):
    #     fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
    #     board = Board(fen)
    #
    #     assert board.pieces_on_light_squares(True) == 4
    #     assert board.pieces_on_light_squares(False) == 3
    #     assert board.pawns_on_light_squares(True) == 2
    #     assert board.pawns_on_light_squares(False) == 5
    #
    #     assert board.pieces_on_dark_squares(True) == 3
    #     assert board.pieces_on_dark_squares(False) == 4
    #     assert board.pawns_on_dark_squares(True) == 5
    #     assert board.pawns_on_dark_squares(False) == 2
    #
    # def test_direct_threats(self):
    #     fen = "8/8/2b5/5q2/3N4/1p6/R1P5/8 w - - 0 1"
    #     board = Board(fen)
    #
    #     assert board.get_direct_threats(True) == 12
    #     assert board.get_direct_threats(False) == 5
    #
    # def test_king_mobility(self):
    #     fen = "k1b5/Bpp5/8/8/8/7P/5Bp1/6K1 w - - 0 1"
    #     board = Board(fen)
    #
    #     assert board.get_king_mobility(True) == 3
    #     assert board.get_king_mobility(False) == 2
    #
    # def test_protection(self):
    #     fen = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
    #     board = Board(fen)
    #
    #     assert board.get_protection(True) == 11
    #     assert board.get_protection(False) == 12
    #
    # def test_as_unique_int(self):
    #     fen = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
    #     board = Board(fen)
    #     pos = board.as_unique_int()
    #     new_board = Board.from_unique_int(pos)
    #
    #     assert board.pawns == new_board.pawns
    #     assert board.knights == new_board.knights
    #     assert board.bishops == new_board.bishops
    #     assert board.rooks == new_board.rooks
    #     assert board.queens == new_board.queens
    #     assert board.kings == new_board.kings
    #     assert board.occupied == new_board.occupied
    #     assert board.occupied_co == new_board.occupied_co
    #
    # def test_load_shm_board_perf(self):
    #     fen = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
    #     board = Board(fen)
    #     mm = MemoryManager()
    #     mm.set_node_board("test", board)
    #
    #     new_board = mm.get_node_board("test")
    #     assert board.pawns == new_board.pawns
    #     assert board.knights == new_board.knights
    #     assert board.bishops == new_board.bishops
    #     assert board.rooks == new_board.rooks
    #     assert board.queens == new_board.queens
    #     assert board.kings == new_board.kings
    #     assert board.occupied == new_board.occupied
    #     assert board.occupied_co == new_board.occupied_co
    #
    #     k = 0
    #     t0 = perf_counter()
    #     while perf_counter() - t0 < 1:
    #         mm.get_node_board("test")
    #         k += 1
    #
    #     mm.remove_node_memory("test")
    #
    #     print(k)
    #     assert k > 50000
    #
    # def test_create_from_position_perf(self):
    #     fen = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
    #     board = Board(fen)
    #     pos = board.as_unique_int()
    #
    #     k = 0
    #     t0 = perf_counter()
    #     while perf_counter() - t0 < 1:
    #         Board.from_unique_int(pos)
    #         k += 1
    #
    #     print(k)
    #     assert k > 240000

    def test_get_king_mobility(self):
        fen = "1rN4r/1p4p1/1n5p/k5p1/2P3Q1/pPp5/1PK5/8 w - - 0 1"
        board = Board(fen)
        assert board.get_king_mobility(True) == 5
        assert board.get_king_mobility(False) == 4

    def test_get_king_proximity_map_normalized(self):
        fen = "1rN4r/1p4p1/1n5p/k5p1/2P3Q1/pPp5/1PK5/8 w - - 0 1"
        board = Board(fen)
        white_proximity = (
            7.0 - board.get_king_proximity_map_normalized(True) * 7.0
        ).tolist()
        black_proximity = (
            7.0 - board.get_king_proximity_map_normalized(False) * 7.0
        ).tolist()
        # fmt: off
        assert [int(k) for k in white_proximity] == [
            2, 1, 1, 1, 2, 3, 4, 5,
            2, 1, 0, 1, 2, 3, 4, 5,
            2, 1, 1, 1, 2, 3, 4, 5,
            2, 2, 2, 2, 2, 3, 4, 5,
            3, 3, 3, 3, 3, 3, 4, 5,
            4, 4, 4, 4, 4, 4, 4, 5,
            5, 5, 5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6, 6, 6
        ]
        assert [int(k) for k in black_proximity] == [
            4, 4, 4, 4, 4, 5, 6, 7,
            3, 3, 3, 3, 4, 5, 6, 7,
            2, 2, 2, 3, 4, 5, 6, 7,
            1, 1, 2, 3, 4, 5, 6, 7,
            0, 1, 2, 3, 4, 5, 6, 7,
            1, 1, 2, 3, 4, 5, 6, 7,
            2, 2, 2, 3, 4, 5, 6, 7,
            3, 3, 3, 3, 4, 5, 6, 7
        ]
        # fmt: on

    @parameterized.expand([
        [
            "1rN4r/1p4p1/1n5p/k5p1/2P3Q1/pPp5/1PK5/8 w - - 0 1",
            # fmt: off
            [
                0, 1, 1, 2, 0, 0, 1, 0,
                0, -1, 0, 0, 1, 0, 1, 0,
                1, 1, 2, 1, 0, 1, 1, 1,
                -1, -1, 1, 1, 1, 0, 0, 0,
                0, 0, 0, 0, 0, 1, 0, 1,
                -2, 0, -1, 1, 1, -1, 0, -2,
                1, -1, 0, 0, 1, 0, 0, -1,
                -2, 0, -2, -1, -1, -1, -1, 0,
            ]  # fmt: on
        ],
        [
            "rnbqkbnr/pp1ppppp/2p5/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            # fmt: off
            [
                0, 1, 1, 1, 1, 1, 1, 0,
                1, 1, 1, 4, 4, 1, 1, 1,
                2, 2, 2, 3, 2, 3, 2, 2,
                0, 0, 0, 1, 0, 1, 0, 0,
                -1, -1, 1, -1, 1, 0, 1, 0,
                -2, -2, -3, -1, -2, -3, -2, -1,
                -1, -1, -1, -4, -4, -1, -1, -1,
                0, -1, -1, -1, -1, -1, -1, 0
            ]  # fmt: on
        ],
    ])
    def test_get_square_control_map_for_both(self, fen, expected_map):
        board = Board(fen)
        map = board.get_square_control_map_for_both().tolist()
        try:
            assert map == expected_map
        except AssertionError:
            print(map)

    def test_get_occupied_square_value_map(self):
        fen = "nR4R1/n1B5/3b2Q1/2kN3r/6bq/2rBK3/7N/8 w - - 0 1"
        board = Board(fen)
        # fmt: off
        assert [int(k) for k in board.get_occupied_square_value_map(True)] == [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 3,
            0, 0, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 9, 0,
            0, 0, 3, 0, 0, 0, 0, 0,
            0, 5, 0, 0, 0, 0, 5, 0,
        ]
        assert [int(k) for k in board.get_occupied_square_value_map(False)] == [
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 5, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 3, 9,
            0, 0, 0, 0, 0, 0, 0, 5,
            0, 0, 0, 3, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0,
            3, 0, 0, 0, 0, 0, 0, 0,
        ]
        # fmt: on

    # def test_eval_perf(self):
    #     fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"  # stockfish +0.6
    #     # fen = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"  # stockfish -0.6
    #     board = Board(fen)
    #     eval = SquareControlEval()
    #
    #     k = 0
    #     t0 = perf_counter()
    #     while perf_counter() - t0 < 1:
    #         eval.get_score(board, "", 0, True)
    #         k += 1
    #
    #     print(eval.get_score(board, "", 0, True))
    #     print(k)
    #     assert k > 12500

    # def test_eval_worker_perf(self):
    #     fen = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
    #     board = Board(fen)
    #     eval_worker = EvalWorker(None, None, 0, True, None)
    #     eval_worker.setup()
    #
    #     k = 0
    #     t0 = perf_counter()
    #     while perf_counter() - t0 < 1:
    #         eval_worker.eval_item(board, ".", "g3d6")
    #         k += 1
    #
    #     print(k)
    #     assert k > 10000

    # def test_get_occupied_square_value_map_for_both(self):
    #     fen = "nR4R1/n1B5/3b2Q1/2kN3r/6bq/2rBK3/7N/8 w - - 0 1"
    #     board = Board(fen)
    #
    #     w = board.get_occupied_square_value_map(True)
    #     b = board.get_occupied_square_value_map(False)
    #
    #     w_, b_ = board.get_occupied_square_value_map_for_both()
    #
    #     assert (w == w_).all()
    #     assert (b == b_).all()
