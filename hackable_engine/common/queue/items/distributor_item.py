# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from struct import pack, unpack
from typing import ClassVar

from numpy import float32

from hackable_engine.board.chess.serializers.chess_board_serializer_mixin import (
    CHESS_BOARD_BYTES_NUMBER,
)
from hackable_engine.common.queue.items.base_item import BaseItem


@dataclass(slots=True)
class DistributorItem(BaseItem):
    """
    Item passed through DistributorQueue.
    """

    board_bytes_number: ClassVar[int] = CHESS_BOARD_BYTES_NUMBER
    run_id: str
    node_name: str
    forcing_level: int
    score: float32
    board: bytes

    def __init__(
        self,
        run_id: str,
        node_name: str,
        forcing_level: int,
        score: float32,
        board: bytes,
    ) -> None:
        self.run_id: str = run_id
        self.node_name: str = node_name
        self.forcing_level: int = forcing_level
        self.score: float32 = score
        self.board: bytes = board

    @classmethod
    def loads(cls, bytes_: bytes) -> DistributorItem:
        """"""

        board_and_float_bytes_number = DistributorItem.board_bytes_number + 4

        string_part = bytes_[:-board_and_float_bytes_number]
        float_part = bytes_[
            -board_and_float_bytes_number : -DistributorItem.board_bytes_number
        ]
        board = bytes_[-DistributorItem.board_bytes_number :]
        values = string_part.decode("utf-8").split(";")

        return DistributorItem(
            values[0],
            values[1],
            int(values[2]),
            unpack("f", float_part)[0],
            board,
        )

    def dumps(self: DistributorItem) -> bytes:
        """"""

        score_bytes = pack("f", self.score)

        return (
            f"{self.run_id};{self.node_name};{self.forcing_level}".encode()
            + score_bytes
            + self.board
        )
