# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Iterator

from numpy import float32

from arek_chess.common.queue.items.base_item import BaseItem


@dataclass
class DistributorItem(BaseItem):
    """
    Item passed through DistributorQueue.
    """

    __slots__ = ("run_id", "node_name", "move_str", "score", "captured")

    node_name: str
    move_str: str
    score: float32
    captured: int

    def __iter__(self) -> Iterator:
        return self.node_name, self.move_str, self.score, self.captured
