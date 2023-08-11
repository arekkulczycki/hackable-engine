# -*- coding: utf-8 -*-
from struct import pack, unpack

from arek_chess.board.hex.mixins import HexBoardProtocol

BOARD_BYTES_NUMBER: int = 17
VEC_1: int = 2**13 - 1


class HexBoardSerializerMixin:
    """
    Serialize positions into byte arrays and back.
    """

    def serialize_position(self: HexBoardProtocol) -> bytes:
        """
        :return: bytes array of length 73
        """

        whites = pack("Q", self.occupied_co[True])
        blacks = pack("Q", self.occupied_co[False])
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

        whites_bytes = bytes_[:8]
        blacks_bytes = bytes_[8:16]
        turn = bytes_[-1:]

        self.occupied_co[True] = unpack("Q", whites_bytes)[0]
        self.occupied_co[False] = unpack("Q", blacks_bytes)[0]
        self.unoccupied = VEC_1 & ~(self.occupied_co[True] | self.occupied_co[False])
        self.turn = unpack("?", turn)[0]
