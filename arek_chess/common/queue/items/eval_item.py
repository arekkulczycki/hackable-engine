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

    def __init__(self, run_id: str, node_name: str, move_str: str) -> None:
        self.run_id: str = run_id
        self.node_name: str = node_name
        self.move_str: str = move_str

    @staticmethod
    def loads(b: memoryview) -> EvalItem:
        """"""

        return EvalItem(*b.tobytes().decode("utf-8").split(";"))

    @staticmethod
    def dumps(obj: EvalItem) -> bytes:
        """"""

        return f"{obj.run_id};{obj.node_name};{obj.move_str}".encode()
