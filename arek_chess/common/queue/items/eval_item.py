# -*- coding: utf-8 -*-.
from __future__ import annotations

from arek_chess.board.chess.mixins.chess_board_serializer_mixin import (
    BOARD_BYTES_NUMBER,
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

    def __init__(
        self, run_id: str, node_name: str, move_str: str, is_forcing: int, board: bytes
    ) -> None:
        self.run_id: str = run_id
        self.node_name: str = node_name
        self.move_str: str = move_str
        self.is_forcing: int = is_forcing
        self.board: bytes = board

    @staticmethod
    def loads(bytes_: bytes) -> EvalItem:
        """"""

        string_part = bytes_[:-BOARD_BYTES_NUMBER]
        board = bytes_[-BOARD_BYTES_NUMBER:]
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
            f"{obj.run_id};{obj.node_name};{obj.move_str};{obj.is_forcing}".encode()
            + obj.board
        )
