# -*- coding: utf-8 -*-
from __future__ import annotations

from hackable_engine.common.queue.items.base_item import BaseItem


# @dataclass
class ControlItem(BaseItem):
    """
    Item passed through ControlQueue.
    """

    def __init__(self, run_id: str, control_value: str) -> None:
        self.run_id: str = run_id
        self.control_value: str = control_value

    @staticmethod
    def loads(bytes_: bytes) -> ControlItem:
        """"""

        return ControlItem(*bytes_.decode("utf-8").split(";"))

    @staticmethod
    def dumps(obj: ControlItem) -> bytes:
        """"""

        return f"{obj.run_id};{obj.control_value}".encode()
