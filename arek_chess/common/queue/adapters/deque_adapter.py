# -*- coding: utf-8 -*-

from collections import deque
from queue import Empty, Full
from typing import Callable, List, Optional

from common.queue.base_queue import BaseQueue
from common.queue.items.base_item import BaseItem


class DequeAdapter(BaseQueue):
    """
    Queue provided by external FasterFifo library.
    """

    def __init__(self, name: str, loader: Callable, dumper: Callable) -> None:
        """
        Initialize a queue of a chosen queuing class.
        """

        super().__init__(name)
        self.queue = deque(maxlen=1024 * 1024 * 10)

    def put(self, item: BaseItem) -> None:
        """"""

        try:
            self.queue.append(item)
        except Full:  # TODO: probably doesn raise this
            raise

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        try:
            self.queue.extend(items)
        except Full:  # TODO: probably doesn raise this
            raise

    def get(self) -> Optional[BaseItem]:
        """"""

        try:
            return self.queue.popleft()
        except Empty:  # TODO: maybe doesn raise this?
            return None

    def get_many(self, max_messages_to_get: int, timeout: float = 0) -> List[BaseItem]:
        """"""

        try:
            return [item for item in (self.queue.popleft() for i in range(max_messages_to_get)) if item is not None]
        except Empty:  # TODO: maybe doesn raise this?
            return []

    def _get_many_blocking(
        self, max_messages_to_get: int, timeout: float
    ) -> List[BaseItem]:
        """"""

        try:
            return self.get_many(max_messages_to_get)
        except Empty:
            return []

    def is_empty(self) -> bool:
        """"""

        return False

    def size(self) -> int:
        """"""

        return self.queue.maxlen

    def close(self) -> None:
        """"""

        self.queue.clear()
