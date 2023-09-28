# -*- coding: utf-8 -*-
from typing import Tuple
from unittest import TestCase

from numpy import asarray, array_equal, int8
from parameterized import parameterized

from arek_chess.board.hex.hex_board import HexBoard, Move

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
    def test_get_connectedness_and_wingspan(
        self, notation: str, expectation: Tuple[int, int, int, int]
    ) -> None:
        board = HexBoard(notation, size=13)

        assert board.get_connectedness_and_wingspan() == expectation

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
            ["", 7, 7],
            ["d4g2b7e4a3c3g1f6b6b4e2d7g5d1c2d5c1g3f4a2c4c5d2g4e6a4g6f2b5a7c7a6d6a5e7b3g7", 2, 1],
        ]
    )
    def test_get_shortest_missing_distance(self, notation, missing_distance_white, missing_distance_black) -> None:
        board = HexBoard(notation, size=7)

        assert board.get_shortest_missing_distance(True) == missing_distance_white
        assert board.get_shortest_missing_distance(False) == missing_distance_black

    @parameterized.expand(
        [
            [
                # fmt: off
                "e5c10i5f7", asarray([
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0, 2],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                ], dtype=int8), 7
            ],
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
        ],
    )
    def test_get_neighbourhood(self, notation, arr, neighbourhood_size) -> None:
        board = HexBoard(notation, size=13)
        board.move_stack.append(Move.from_coord(notation[-2:], size=13))

        assert array_equal(board.get_neighbourhood(neighbourhood_size), arr.flatten())
