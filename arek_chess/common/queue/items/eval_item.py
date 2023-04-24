# -*- coding: utf-8 -*-.

from dataclasses import dataclass
from typing import Iterator

from arek_chess.common.queue.items.base_item import BaseItem


@dataclass
class EvalItem(BaseItem):
    """
    Item passed through EvalQueue.
    """

    __slots__ = ("run_id", "node_name", "move_str")

    node_name: str
    move_str: str

    def __iter__(self) -> Iterator[str]:
        return self.node_name, self.move_str
