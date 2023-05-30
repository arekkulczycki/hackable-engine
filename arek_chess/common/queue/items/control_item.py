# -*- coding: utf-8 -*-
from __future__ import annotations

from arek_chess.common.queue.items.base_item import BaseItem


# @dataclass
class ControlItem(BaseItem):
    """
    Item passed through ControlQueue.
    """

    # __slots__ = ("run_id", "control_value")

    # control_value: str

    def __init__(self, run_id: str, control_value: str) -> None:
        self.run_id: str = run_id
        self.control_value: str = control_value

    @staticmethod
    def loads(b: memoryview) -> ControlItem:
        """"""

        return ControlItem(*b.tobytes().decode("utf-8").split(";"))

    @staticmethod
    def dumps(obj: ControlItem) -> bytes:
        """"""

        return f"{obj.run_id};{obj.control_value}".encode()
