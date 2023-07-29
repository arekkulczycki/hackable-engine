# -*- coding: utf-8 -*-
from __future__ import annotations

from struct import pack, unpack

from numpy import float32

from arek_chess.board.mixins.board_serializer_mixin import BOARD_BYTES_NUMBER
from arek_chess.common.queue.items.base_item import BaseItem

BOARD_AND_FLOAT_BYTES_NUMBER = BOARD_BYTES_NUMBER + 4


# @dataclass
class DistributorItem(BaseItem):
    """
    Item passed through DistributorQueue.
    """

    # __slots__ = ("run_id", "node_name", "move_str", "score", "captured")

    # node_name: str
    # move_str: str
    # captured: int
    # score: float32
    # board: bytes

    def __init__(
        self,
        run_id: str,
        node_name: str,
        move_str: str,
        captured: int,
        score: float32,
        board: bytes,
    ) -> None:
        self.run_id: str = run_id
        self.node_name: str = node_name
        self.move_str: str = move_str
        self.score: float32 = score
        self.captured: int = captured
        self.board: bytes = board

    @staticmethod
    def loads(bytes_: bytes) -> DistributorItem:
        """"""

        string_part = bytes_[:-BOARD_AND_FLOAT_BYTES_NUMBER]
        float_part = bytes_[-BOARD_AND_FLOAT_BYTES_NUMBER:-BOARD_BYTES_NUMBER]
        board = bytes_[-BOARD_BYTES_NUMBER:]
        values = string_part.decode("utf-8").split(";")

        return DistributorItem(
            values[0],
            values[1],
            values[2],
            int(values[3]),
            unpack("f", float_part)[0],
            board,
        )

    @staticmethod
    def dumps(obj: DistributorItem) -> bytes:
        """"""

        score_bytes = pack("f", obj.score)

        return (
            f"{obj.run_id};{obj.node_name};{obj.move_str};{obj.captured}".encode()
            + score_bytes
            + obj.board
        )
