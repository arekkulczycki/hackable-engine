# -*- coding: utf-8 -*-
# type: ignore
from unittest import TestCase
from parameterized import parameterized

from hackable_engine.board.hex.hex_board import HexBoard


class HexBoardSerializerMixinTestCase(TestCase):
    @parameterized.expand(
        [
            [""],
            ["d4g2b7e4a3c3g1f6b6b4e2d7g5d1c2d5c1g3f4a2c4c5d2g4e6a4g6f2b5a7c7a6d6a5e7b3g7"],
        ]
    )
    def test_serialize_board_7(self, notation: str) -> None:
        """"""

        hb = HexBoard(notation, size=7)

        bytes_ = hb.serialize_position()
        new_hb = HexBoard(size=7)
        new_hb.deserialize_position(bytes_)

        assert hb.occupied_co == new_hb.occupied_co
        assert hb.turn == new_hb.turn
        assert hb.unoccupied == new_hb.unoccupied

    @parameterized.expand(
        [
            [""],
            ["d4g2b7e4a3c3g1f6b6b4e2d7g5d1c2d5c1g3f4a2c4c5d2g4e6a4g6f2b5a7c7a6d6a5e7b3g7"],
        ]
    )
    def test_serialize_board_9(self, notation: str) -> None:
        """"""

        hb = HexBoard(notation, size=9)

        bytes_ = hb.serialize_position()
        new_hb = HexBoard(size=9)
        new_hb.deserialize_position(bytes_)

        assert hb.occupied_co == new_hb.occupied_co
        assert hb.turn == new_hb.turn
        assert hb.unoccupied == new_hb.unoccupied

    @parameterized.expand(
        [
            [""],
            ["d4g2b7e4a3c3g1f6b6b4e2d7g5d1c2d5c1g3f4a2c4c5d2g4e6a4g6f2b5a7c7a6d6a5e7b3g7"],
            ["a8h1b8h2c8h3d8h4e8h5f8h6g8h7b5c2e3b7c7f2e2c6g11i11l11h13k13l13"],
        ]
    )
    def test_serialize_board_13(self, notation: str) -> None:
        """"""

        hb = HexBoard(notation, size=13)

        bytes_ = hb.serialize_position()
        new_hb = HexBoard(size=13)
        new_hb.deserialize_position(bytes_)

        assert hb.occupied_co == new_hb.occupied_co
        assert hb.turn == new_hb.turn
        assert hb.unoccupied == new_hb.unoccupied
