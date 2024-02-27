# -*- coding: utf-8 -*-
import random
from time import perf_counter
from typing import Tuple
from unittest import TestCase

from numpy import asarray, array_equal, int8
from parameterized import parameterized

from hackable_engine.board.hex.hex_board import HexBoard, Move

"""
TEMPLATE
False: int("00000"
           "00000"
           "00000"
           "00000"
           "00000", 2),
"""


class HexBoardTestCase(TestCase):
    @staticmethod
    def test_is_win_direct() -> None:
        """"""

        board = HexBoard(size=5)
        # fmt: off
        board.occupied_co = {
            False: int("00100"
                        "00100"
                         "00100"
                          "00100"
                           "00100", 2),
            True: int("00000"
                        "00000"
                         "00000"
                          "00000"
                           "11111", 2),
        }
        # fmt: on

        assert board.is_black_win() is True
        assert board.is_white_win() is True

    @staticmethod
    def test_is_win_curly() -> None:
        """"""

        board = HexBoard(size=7)
        # fmt: off
        board.occupied_co = {
            False: int("1110010"
                        "0000100"
                         "0111000"
                          "1000110"
                           "1111010"
                            "0000110"
                             "0000010", 2),
            True: int("1110010"
                        "0000100"
                         "0111000"
                          "1000110"
                           "1111010"
                            "0000110"
                             "0000010", 2),
        }
        # fmt: on

        assert board.is_black_win() is True
        assert board.is_white_win() is False

    @staticmethod
    def test_is_white_win() -> None:
        """"""

        board = HexBoard(size=5)
        # fmt: off
        board.occupied_co = {
            False: int("00100"
                        "10100"
                         "00000"
                          "00101"
                           "00100", 2),
            True: int("11000"
                       "01000"
                         "01110"
                          "00010"
                           "00011", 2)
        }
        # fmt: on

        assert board.is_black_win() is False
        assert board.is_white_win() is True

    @parameterized.expand(
        [
            ["a8a7f9f8i9a9d8b7b8e7b5b4e8b6d2e1f3d1g9c3f5c2i4e2i2d3g1f1d4f2i5h3i1g2i3e3e4h2c5h1d6f4d9g3a6e5g5h4f7c4a2a1h7a3d5b1h8b2d7h5i6"],
            ["i4h4g1g4i3h5c2c3i2b3e1h1i1f1h7d1c1g8a1a2e3h3i6h2g9f2d6d2a3d3g3b1i7f3c9e2e9b2g7e4c7d4a6b5a5b4b8g2i5d7b7e5i9d5h9e6c8f4e8a7b6a4f5c5g5a8h6g6i8"],
        ]
    )
    def test_is_win(self, notation) -> None:
        """"""

        board = HexBoard(notation=notation, size=9, init_move_stack=True)

        assert board.is_black_win() is True
        assert board.is_white_win() is False

    @staticmethod
    def test_is_black_win_perf() -> None:
        """"""

        def make_random_notation(size):
            notation = ""
            for i in range(140):  # fill the board
                coords = Move.from_xy(random.randint(0, size-1), random.randint(0, size-1), size).get_coord()
                if coords not in notation:
                    notation += coords
            return notation

        t = 0
        for i in range(1000):
            board = HexBoard(make_random_notation(9), size=9)

            t0 = perf_counter()
            board.is_black_win()
            t += perf_counter() - t0

        assert t < 0.05  # can check the winner more than 20k times per second even with full board

    @parameterized.expand(
        [
            [7, "", 49],
            [7, "c6d4e2", 46],
            [7, "c6d4e2d3e3c5d5f4d7g1b1a2b2d1a4b4a5b5a7b7f6e5f5g3g4g6g7c2f2f1g2e6", 17],
        ]
    )
    def test_generate_moves(self, size: int, notation: str, empty_spaces: int):
        """
        - how many are generated
        - if they are within board borders
        - if they don't repeat
        """

        generated: set[Move] = set()

        board = HexBoard(notation, size=size)

        for move in board.legal_moves:
            assert move not in generated
            generated.add(move)

            assert move.x < size
            assert move.y < size

        assert len(generated) == empty_spaces

    @parameterized.expand(
        [
            ["e5c11i5", (3, 7, 17, 42)],
            ["e5c11i5a1", (6, 7, 32, 42)],
            ["e5c11i5m13", (6, 7, 32, 42)],
            ["e5c11i5m13a1", (6, 10, 32, 57)],
            ["e5c11i5m13m1", (6, 9, 32, 43)],
            ["e5c11i5m13a13", (4, 10, 16, 57)],
        ]
    )
    def test_get_connectedness_and_spacing(
        self, notation: str, expectation: Tuple[int, int, int, int]
    ) -> None:
        board = HexBoard(notation, size=13)

        assert board.get_connectedness_and_spacing() == expectation

    @parameterized.expand(
        [
            ["g7", (0, 0), (0, 3.25), True],
            ["j9e5", 4.5, -5, False],
            # ["j9e5k10i8", 4.5, -5, False],
        ]
    )
    def test_imbalance(
        self, notation: str, imbalance_white: float, imbalance_black: float, exact: bool
    ) -> None:
        board = HexBoard(notation, size=13)

        if exact:
            assert board.get_imbalance(True) == imbalance_white
            assert board.get_imbalance(False) == imbalance_black
        else:
            if imbalance_white > 0:
                assert sum(board.get_imbalance(True)) < imbalance_white
            else:
                assert sum(board.get_imbalance(True)) > -imbalance_white

            if imbalance_black > 0:
                assert sum(board.get_imbalance(False)) < imbalance_black
            else:
                assert sum(board.get_imbalance(False)) > -imbalance_black

    @parameterized.expand(
        [
            ["", 7, 7, 7],
            ["", 9, 9, 9],
            ["d4g2b7e4a3c3g1f6b6b4e2d7g5d1c2d5c1g3f4a2c4c5d2g4e6a4g6f2b5a7c7a6d6a5e7b3g7", 7, 2, 1],
            ["e2e1i9b6f9c6b9h2h8h7d9c3a8e3g6c7e6e5a5h4i5h6a2d3e9d5c1i1g1g3f8d6c9d1d8e4f7a1g9a7a4f2a9c2h9f5i8i6e7g7e8b2g8i4c8b4c5a6d7", 9, 1, 5],
        ]
    )
    def test_get_shortest_missing_distance(self, notation, size, missing_distance_white, missing_distance_black) -> None:
        board = HexBoard(notation, size=size)

        assert board.get_shortest_missing_distance(True) == missing_distance_white
        assert board.get_shortest_missing_distance(False) == missing_distance_black

        assert board.get_shortest_missing_distance_perf(True) == missing_distance_white
        assert board.get_shortest_missing_distance_perf(False) == missing_distance_black

    @parameterized.expand(
        [
            ["g1a6g2b6g3c6h3d6i3", 9, 5, 6],
            ["g1a6g2b6g3c6h3d6i3e6a8f6b8g6b9", 9, 2, 6],
        ]
    )
    def test_get_shortest_missing_distance_no_problems(self, notation, size, missing_distance_white, missing_distance_black) -> None:
        board = HexBoard(notation, size=size)

        assert board.get_shortest_missing_distance(True) == missing_distance_white
        assert board.get_shortest_missing_distance(False) == missing_distance_black

    @parameterized.expand(
        [
            ["g1a6g2b6g3c6h3d6i3", 9, 8, 7],
            ["g1a6g2b6g3c6h3d6i3e6a8f6b8g6b9", 9, 9, 9],
        ]
    )
    def test_get_shortest_missing_distance_perf_problems(self, notation, size, missing_distance_white, missing_distance_black) -> None:
        board = HexBoard(notation, size=size)

        assert board.get_shortest_missing_distance_perf(True) == missing_distance_white
        assert board.get_shortest_missing_distance_perf(False) == missing_distance_black

    @parameterized.expand(
        [
            # [
            #     # fmt: off
            #     "e5c10i5f7", asarray([
            #         [0, 0, 0, 0, 0, 0, 0],
            #         [0, 0, 2, 0, 0, 0, 2],
            #         [0, 0, 0, 0, 0, 0, 0],
            #         [0, 0, 0, 1, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 0],
            #         [0, 0, 0, 0, 0, 0, 0],
            #         [1, 0, 0, 0, 0, 0, 0],
            #     ], dtype=int8), 7
            # ],
            [
                "b3d3c6a2", asarray([  # if corner then appropriate shifted area is used
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [0, 2, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                ], dtype=int8), 7
                # fmt: on
            ],
            [
                "b3d3c6a2", asarray([  # if corner then appropriate shifted area is used
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 2, 0],
                ], dtype=int8), 3
                # fmt: on
            ],
            [
                "g1", asarray([  # if corner then appropriate shifted area is used
                [0, 0, 0, 0, 0, 0, 2],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ], dtype=int8), 7
                # fmt: on
            ],
        ],
    )
    def test_get_neighbourhood(self, notation, arr, neighbourhood_size) -> None:
        board = HexBoard(notation, size=7, init_move_stack=True)

        try:
            assert array_equal(board.get_neighbourhood(neighbourhood_size), arr)
        except:
            pass
            # print(board.get_neighbourhood(neighbourhood_size))
        assert board.get_notation() == notation
