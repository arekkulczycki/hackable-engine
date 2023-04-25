# -*- coding: utf-8 -*-

from __future__ import annotations

from struct import pack, unpack

from numpy import float32

from arek_chess.common.queue.items.base_item import BaseItem


# @dataclass
class SelectorItem(BaseItem):
    """
    Item passed through SelectorQueue.
    """

    # __slots__ = ("run_id", "node_name", "move_str", "score", "captured")

    # node_name: str
    # move_str: str
    # score: float32
    # captured: int

    def __init__(self, node_name: str, move_str: str, score: float32, captured: int) -> None:
        self.node_name: str = node_name
        self.move_str: str = move_str
        self.score: float32 = score
        self.captured: int = captured

    @staticmethod
    def loads(b: memoryview) -> SelectorItem:
        """"""

        string_part, float_part = b.tobytes().split(b"@@@@@")
        values = string_part.decode().split(";")

        return SelectorItem(
            values[0], values[1], unpack("f", float_part)[0], int(values[2])
        )

    @staticmethod
    def dumps(obj: SelectorItem) -> bytes:
        """"""

        score_bytes = pack("f", obj.score)

        return (
            f"{obj.node_name};{obj.move_str};{obj.captured}@@@@@".encode() + score_bytes
        )
