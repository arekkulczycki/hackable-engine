# -*- coding: utf-8 -*-.

from __future__ import annotations

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
        self, run_id: str, node_name: str, move_str: str, captured: int, board: bytes
    ) -> None:
        self.run_id: str = run_id
        self.node_name: str = node_name
        self.move_str: str = move_str
        self.captured: int = captured
        self.board: bytes = board

    @staticmethod
    def loads(b: memoryview) -> EvalItem:
        """"""

        _bytes = b.tobytes()
        string_part = _bytes[:-73]
        board = _bytes[-73:]
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
            f"{obj.run_id};{obj.node_name};{obj.move_str};{obj.captured}".encode()
            + obj.board
        )
