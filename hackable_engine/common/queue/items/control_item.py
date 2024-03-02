# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from hackable_engine.common.queue.items.base_item import BaseItem


@dataclass(slots=True)
class ControlItem(BaseItem):
    """
    Item passed through ControlQueue.
    """

    run_id: str
    control_value: str

    def __init__(self, run_id: str, control_value: str) -> None:
        self.run_id: str = run_id
        self.control_value: str = control_value

    @classmethod
    def loads(cls, bytes_: bytes) -> Self:
        """"""

        return ControlItem(*bytes_.decode("utf-8").split(";"))

    def dumps(self: Self) -> bytes:
        """"""

        return f"{self.run_id};{self.control_value}".encode()
