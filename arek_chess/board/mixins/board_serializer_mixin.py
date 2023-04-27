# -*- coding: utf-8 -*-

from struct import pack, unpack

from arek_chess.board.mixins import BoardProtocol


class BoardSerializerMixin:
    """
    Serialize positions into byte arrays and back.
    """

    def serialize_position(self: BoardProtocol) -> bytes:
        """"""

        pawns = pack("Q", self.pawns)
        knights = pack("Q", self.knights)
        bishops = pack("Q", self.bishops)
        rooks = pack("Q", self.rooks)
        queens = pack("Q", self.queens)
        kings = pack("Q", self.kings)
        whites = pack("Q", self.occupied_co[True])
        blacks = pack("Q", self.occupied_co[False])
        castling_rights = pack("Q", self.castling_rights)
        turn = pack("?", self.turn)

        return b"".join((pawns, knights, bishops, rooks, queens, kings, whites, blacks, castling_rights, turn))

    def deserialize_position(self: BoardProtocol, bytes_: bytes) -> None:
        pawns_bytes = bytes_[:8]
        knights_bytes = bytes_[8:16]
        bishops_bytes = bytes_[16:24]
        rooks_bytes = bytes_[24:32]
        queens_bytes = bytes_[32:40]
        kings_bytes = bytes_[40:48]
        whites_bytes = bytes_[48:56]
        blacks_bytes = bytes_[56:64]
        castling_rights_bytes = bytes_[64:72]
        turn = bytes_[-1:]

        self.pawns = unpack("Q", pawns_bytes)[0]
        self.knights = unpack("Q", knights_bytes)[0]
        self.bishops = unpack("Q", bishops_bytes)[0]
        self.rooks = unpack("Q", rooks_bytes)[0]
        self.queens = unpack("Q", queens_bytes)[0]
        self.kings = unpack("Q", kings_bytes)[0]
        self.occupied_co[True] = unpack("Q", whites_bytes)[0]
        self.occupied_co[False] = unpack("Q", blacks_bytes)[0]
        self.occupied = self.occupied_co[True] | self.occupied_co[False]
        self.castling_rights = unpack("Q", castling_rights_bytes)[0]
        self.turn = unpack("?", turn)[0]
