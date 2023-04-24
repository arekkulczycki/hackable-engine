# -*- coding: utf-8 -*-

from typing import List, Optional

from arek_chess.common.queue.faster_fifo_queue import FasterFifoQueue
from arek_chess.common.queue.items.base_item import BaseItem


class QueueManager:
    """
    Class_docstring
    """

    def __init__(self, name: str):
        """
        Initialize a queue of a chosen queuing class.
        """

        self.queue = FasterFifoQueue(name)
        # self.queue = RedisQueue(name)
        # self.queue = RabbitmqQueue(name)

    @property
    def name(self) -> str:
        return self.queue.name

    def put(self, item: BaseItem) -> None:
        """"""

        self.queue.put(item)

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        self.queue.put_many(items)

    def get(self) -> Optional[BaseItem]:
        """"""

        return self.queue.get()

    def get_many(
        self, max_messages_to_get: int = 10, timeout: int = 0
    ) -> List[BaseItem]:
        """"""

        return self.queue.get_many(max_messages_to_get, timeout)

    def is_empty(self) -> bool:
        """"""

        return self.queue.is_empty()

    def size(self) -> int:
        """"""

        return self.queue.size()

    def close(self) -> None:
        """"""

        self.queue.close()