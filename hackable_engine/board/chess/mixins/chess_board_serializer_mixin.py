# -*- coding: utf-8 -*-
from struct import pack, unpack
from typing import Optional

from hackable_engine.board.chess.mixins import ChessBoardProtocol

CHESS_BOARD_BYTES_NUMBER = 75
NONE_EP_SQUARE = 64


class ChessBoardSerializerMixin:
    """
    Serialize positions into byte arrays and back.
    """

    def serialize_position(self: ChessBoardProtocol) -> bytes:
        """
        :return: bytes array of length 75
        """

        ep_square_: Optional[int] = self.ep_square

        pawns = pack("Q", self.pawns)
        knights = pack("Q", self.knights)
        bishops = pack("Q", self.bishops)
        rooks = pack("Q", self.rooks)
        queens = pack("Q", self.queens)
        kings = pack("Q", self.kings)
        whites = pack("Q", self.occupied_co[True])
        blacks = pack("Q", self.occupied_co[False])
        castling_rights = pack("Q", self.castling_rights)
        ep_square = pack("H", ep_square_ if ep_square_ is not None else NONE_EP_SQUARE)
        turn = pack("?", self.turn)

        return b"".join(
            (
                pawns,
                knights,
                bishops,
                rooks,
                queens,
                kings,
                whites,
                blacks,
                castling_rights,
                ep_square,
                turn,
            )
        )

    def deserialize_position(self: ChessBoardProtocol, bytes_: bytes) -> None:
        """"""

        pawns_bytes = bytes_[:8]
        knights_bytes = bytes_[8:16]
        bishops_bytes = bytes_[16:24]
        rooks_bytes = bytes_[24:32]
        queens_bytes = bytes_[32:40]
        kings_bytes = bytes_[40:48]
        whites_bytes = bytes_[48:56]
        blacks_bytes = bytes_[56:64]
        castling_rights_bytes = bytes_[64:72]
        ep_square = bytes_[72:74]
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
        ep_square_ = unpack("H", ep_square)[0]
        self.ep_square = ep_square_ if ep_square_ < 64 else None
        self.turn = unpack("?", turn)[0]
