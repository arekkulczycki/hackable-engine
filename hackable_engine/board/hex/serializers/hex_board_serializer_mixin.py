# -*- coding: utf-8 -*-
from struct import pack, unpack
from typing import Tuple

from hackable_engine.board import BitBoard
from hackable_engine.board.hex.serializers import HexBoardProtocol


class HexBoardSerializerMixin:
    """
    Serialize positions into byte arrays and back.
    """

    @property
    def board_bytes_number(self: HexBoardProtocol) -> int:
        """"""

        return self._board_bytes_number(self.size_square)

    @staticmethod
    def _board_bytes_number(size_square: int) -> int:
        """"""

        long_size: int = 8
        color_sets: int = 2
        turn_bytes: int = 1

        return color_sets * long_size * ((size_square - 1) // long_size ** 2 + 1) + turn_bytes

    def serialize_position(self: HexBoardProtocol) -> bytes:
        """
        :return: bytes array of length 17
        """

        whites = b"".join(
            [pack("Q", cmp) for cmp in self.get_occupied_in_components(True)]
        )
        blacks = b"".join(
            [pack("Q", cmp) for cmp in self.get_occupied_in_components(False)]
        )
        turn = pack("?", self.turn)

        return b"".join(
            (
                whites,
                blacks,
                turn,
            )
        )

    def deserialize_position(self: HexBoardProtocol, bytes_: bytes) -> None:
        """"""

        long_size = 8
        bytes_number = long_size * ((self.size_square - 1) // long_size ** 2 + 1)

        whites_bytes = bytes_[:bytes_number]
        blacks_bytes = bytes_[bytes_number : 2 * bytes_number]
        turn = bytes_[-1:]

        self.occupied_co[True] = self.unpack_components(whites_bytes)
        self.occupied_co[False] = self.unpack_components(blacks_bytes)
        self.unoccupied = self.get_all_mask() ^ (
            self.occupied_co[True] | self.occupied_co[False]
        )

        self.turn = unpack("?", turn)[0]

    def get_occupied_in_components(self: HexBoardProtocol, color: bool) -> Tuple[BitBoard, ...]:
        """"""

        if self.size < 9:
            return (self.occupied_co[color],)

        if self.size < 12:
            return self.split_bitboard_in_two_components(self.occupied_co[color])

        if self.size < 14:
            return self.split_bitboard_in_three_components(self.occupied_co[color])

        raise NotImplementedError("Larger than 24x24 board size not implemented")

    @staticmethod
    def split_bitboard_in_two_components(b: BitBoard) -> Tuple[BitBoard, BitBoard]:
        """
        :returns: last (right) N bytes and first (left) N (or N+1) bytes
        """

        one = (1 << 64) - 1
        return b & one, b >> 64

    @staticmethod
    def split_bitboard_in_three_components(b: BitBoard) -> Tuple[BitBoard, BitBoard, BitBoard]:
        """
        :returns: last (right) N bytes, middle N bytes and first (left) N (or N+1) bytes
        """

        one = (1 << 64) - 1
        return b & one, (b >> 64) & one, b >> (2 * 64)

    def unpack_components(self: HexBoardProtocol, bytes_: bytes) -> BitBoard:
        """"""

        if self.size < 9:
            return unpack("Q", bytes_)[0]

        if self.size < 12:
            return unpack("Q", bytes_[:8])[0] + (unpack("Q", bytes_[8:])[0] << 64)

        if self.size < 14:
            return (
                unpack("Q", bytes_[:8])[0]
                + (unpack("Q", bytes_[8:16])[0] << 64)
                + (unpack("Q", bytes_[16:])[0] << 2 * 64)
            )

        raise NotImplementedError("Larger than 24x24 board size not implemented")
