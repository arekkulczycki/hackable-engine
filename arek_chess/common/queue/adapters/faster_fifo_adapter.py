# -*- coding: utf-8 -*-
from functools import partial
from queue import Empty, Full
from typing import Callable, List, Optional

from faster_fifo import Queue

from arek_chess.common.queue.base_queue import BaseQueue
from arek_chess.common.queue.items.base_item import BaseItem
# from larch.pickle.pickle import dumps, loads


class FasterFifoAdapter(BaseQueue):
    """
    Queue provided by external FasterFifo library.
    """

    def __init__(
        self,
        name: str,
        loader: Optional[Callable] = None,
        dumper: Optional[Callable] = None,
    ):
        """
        Initialize a faster-fifo queue with given (de)serializers.
        """

        super().__init__(name)
        self.queue = Queue(
            max_size_bytes=1024 * 1024 * 100,
            loads=partial(self.memoryview_loader, loader),
            dumps=dumper,
            # loads=partial(self.memoryview_loader, loader)
            # if loader
            # else (lambda _bytes: loads(_bytes.tobytes())),
            # dumps=dumper or partial(dumps, protocol=5, with_refs=False),
        )

    @staticmethod
    def memoryview_loader(loader: Callable, memview: memoryview) -> BaseItem:
        item: BaseItem = loader(memview.tobytes())
        memview.release()
        return item

    @staticmethod
    def memoryview_dumper(dumper: Callable, memview: memoryview) -> Callable:
        return dumper(memview.tobytes())

    def put(self, item: BaseItem) -> None:
        """"""

        try:
            self.queue.put(item)
        except Full:
            raise

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        try:
            self.queue.put_many_nowait(items)
        except Full:
            raise

    def get(self) -> Optional[BaseItem]:
        """"""

        try:
            return self.queue.get_nowait()
        except Empty:
            return None

    def get_many(self, max_messages_to_get: int, timeout: float = 0) -> List[BaseItem]:
        """"""

        if timeout:
            return self._get_many_blocking(max_messages_to_get, timeout)

        try:
            return self.queue.get_many_nowait(max_messages_to_get=max_messages_to_get)
        except Empty:
            return []

    def _get_many_blocking(
        self, max_messages_to_get: int, timeout: float
    ) -> List[BaseItem]:
        """"""

        try:
            return self.queue.get_many(
                max_messages_to_get=max_messages_to_get, block=True, timeout=timeout
            )
        except Empty:
            return []

    def get_all(self) -> List[BaseItem]:
        """"""

        try:
            return self.queue.get_many()
        except Empty:
            return []

    def is_empty(self) -> bool:
        """"""

        return self.queue.empty()

    def size(self) -> int:
        """"""

        return self.queue.qsize()

    def close(self) -> None:
        """"""

        self.queue.close()
