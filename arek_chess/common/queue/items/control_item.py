# -*- coding: utf-8 -*-

from dataclasses import dataclass

from arek_chess.common.queue.items.base_item import BaseItem


@dataclass
class ControlItem(BaseItem):
    """
    Item passed through ControlQueue.
    """

    __slots__ = ("run_id", "control_value")

    control_value: str
