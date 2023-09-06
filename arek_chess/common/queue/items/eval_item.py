# -*- coding: utf-8 -*-.
from __future__ import annotations

from typing import ClassVar

from arek_chess.board.chess.mixins.chess_board_serializer_mixin import (
    CHESS_BOARD_BYTES_NUMBER,
)
from arek_chess.common.queue.items.base_item import BaseItem


# @dataclass
class EvalItem(BaseItem):
    """
    Item passed through EvalQueue.
    """

    # __slots__ = ("run_id", "node_name", "move_str")

    # node_name: str
    # move_str: str
    # captured: int
    # board: bytes
    board_bytes_number: ClassVar[int] = CHESS_BOARD_BYTES_NUMBER

    def __init__(
        self, run_id: str, parent_node_name: str, move_str: str, forcing_level: int, board: bytes
    ) -> None:
        self.run_id: str = run_id
        self.parent_node_name: str = parent_node_name
        self.move_str: str = move_str
        self.forcing_level: int = forcing_level
        self.board: bytes = board

    @staticmethod
    def loads(bytes_: bytes) -> EvalItem:
        """"""

        string_part = bytes_[:-EvalItem.board_bytes_number]
        board = bytes_[-EvalItem.board_bytes_number:]
        values = string_part.decode("utf-8").split(";")

        return EvalItem(
            values[0],
            values[1],
            values[2],
            int(values[3]),
            board,
        )

    @staticmethod
    def dumps(obj: EvalItem) -> bytes:
        """"""

        return (
            f"{obj.run_id};{obj.parent_node_name};{obj.move_str};{obj.forcing_level}".encode()
            + obj.board
        )
