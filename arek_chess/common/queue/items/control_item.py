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

    def __init__(self, control_value: str) -> None:
        self.control_value: str = control_value

    @staticmethod
    def loads(b: memoryview) -> ControlItem:
        """"""

        return ControlItem(b.tobytes().decode("utf-8"))

    @staticmethod
    def dumps(obj: ControlItem) -> bytes:
        """"""

        return obj.control_value.encode()
