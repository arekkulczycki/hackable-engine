# -*- coding: utf-8 -*-

from __future__ import annotations

from struct import pack, unpack

from numpy import float32

from arek_chess.common.queue.items.base_item import BaseItem


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
    def loads(b: memoryview) -> DistributorItem:
        """"""

        _bytes = b.tobytes()
        string_part = _bytes[:-77]
        float_part = _bytes[-77:-73]
        board = _bytes[-73:]
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
