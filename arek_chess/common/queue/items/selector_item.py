# -*- coding: utf-8 -*-
from __future__ import annotations

from struct import pack, unpack
from typing import ClassVar

from numpy import float32

from arek_chess.board.chess.mixins.chess_board_serializer_mixin import (
    CHESS_BOARD_BYTES_NUMBER,
)
from arek_chess.common.queue.items.base_item import BaseItem


# @dataclass
class SelectorItem(BaseItem):
    """
    Item passed through SelectorQueue.
    """

    # __slots__ = ("run_id", "node_name", "move_str", "score", "captured")

    # node_name: str
    # move_str: str
    # captured: int
    # score: float32
    # board: bytes
    board_bytes_number: ClassVar[int] = CHESS_BOARD_BYTES_NUMBER

    def __init__(
        self,
        run_id: str,
        parent_node_name: str,
        move_str: str,
        forcing_level: int,
        score: float32,
        board: bytes,
    ) -> None:
        self.run_id: str = run_id
        self.parent_node_name: str = parent_node_name
        self.move_str: str = move_str
        self.forcing_level: int = forcing_level
        self.score: float32 = score
        self.board: bytes = board

    @staticmethod
    def loads(bytes_: bytes) -> SelectorItem:
        """"""

        board_and_float_bytes_number = SelectorItem.board_bytes_number + 4

        string_part = bytes_[:-board_and_float_bytes_number]
        float_part = bytes_[-board_and_float_bytes_number:-SelectorItem.board_bytes_number]
        board = bytes_[-SelectorItem.board_bytes_number:]
        values = string_part.decode("utf-8").split(";")

        return SelectorItem(
            values[0],
            values[1],
            values[2],
            int(values[3]),
            unpack("f", float_part)[0],
            board,
        )

    @staticmethod
    def dumps(obj: SelectorItem) -> bytes:
        """"""

        score_bytes = pack("f", obj.score)

        return (
            f"{obj.run_id};{obj.parent_node_name};{obj.move_str};{obj.forcing_level}".encode()
            + score_bytes
            + obj.board
        )
