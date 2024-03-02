# -*- coding: utf-8 -*-.
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
class EvalItem(BaseItem):
    """
    Item passed through EvalQueue.
    """

    board_bytes_number: ClassVar[int] = CHESS_BOARD_BYTES_NUMBER
    run_id: str
    parent_node_name: str
    move_str: str
    forcing_level: int
    model_score: float32
    board: bytes

    def __init__(
        self,
        run_id: str,
        parent_node_name: str,
        move_str: str,
        forcing_level: int,
        model_score: float32,
        board: bytes,
    ) -> None:
        self.run_id: str = run_id
        self.parent_node_name: str = parent_node_name
        self.move_str: str = move_str
        self.forcing_level: int = forcing_level
        self.model_score: float32 = model_score
        self.board: bytes = board

    @classmethod
    def loads(cls, bytes_: bytes) -> EvalItem:
        """"""

        board_and_float_bytes_number = EvalItem.board_bytes_number + 4

        string_part = bytes_[:-board_and_float_bytes_number]
        float_part = bytes_[
            -board_and_float_bytes_number : -EvalItem.board_bytes_number
        ]
        board = bytes_[-EvalItem.board_bytes_number :]
        values = string_part.decode("utf-8").split(";")

        return EvalItem(
            values[0],
            values[1],
            values[2],
            int(values[3]),
            unpack("f", float_part)[0],
            board,
        )

    def dumps(self: EvalItem) -> bytes:
        """"""

        score_bytes = pack("f", self.model_score)

        return (
            f"{self.run_id};{self.parent_node_name};{self.move_str};{self.forcing_level}".encode()
            + score_bytes
            + self.board
        )
