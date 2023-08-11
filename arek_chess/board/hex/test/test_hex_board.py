# -*- coding: utf-8 -*-
from unittest import TestCase
from pytest import mark

from arek_chess.board.hex.hex_board import HexBoard

"""
TEMPLATE
False: int("00000"
           "00000"
           "00000"
           "00000"
           "00000", 2),
"""


class HexBoardTestCase(TestCase):
    """
    Class_docstring
    """

    @mark.parameterized
    def test_is_win_direct(self) -> None:
        """"""

        board = HexBoard(5)
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

    @mark.parameterized
    def test_is_win_curly(self) -> None:
        """"""

        board = HexBoard(7)
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

    @mark.parameterized
    def test_is_white_win(self) -> None:
        """"""

        board = HexBoard(5)
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
