# -*- coding: utf-8 -*-
import math
from collections import deque
from random import choice
from typing import Callable, List, Optional

from js import postMessage
from pyodide.ffi import to_js

from hackable_engine.common.queue.base_queue import BaseQueue
from hackable_engine.common.queue.items.base_item import BaseItem


class WasmAdapter(BaseQueue):
    """
    Queue based on python deque, but receiving and publishing items via wasm module provided by Pyodide.
    """

    def __init__(
        self,
        name: str,
        loader: Optional[Callable] = None,
        dumper: Optional[Callable] = None,
    ):
        """
        Initialize a wasm queue with given (de)serializers.

        Built upon a DequeAdapter, although when an item is put on this queue it's propagated out to JS.
        """

        super().__init__(name)

        self.loads: Callable = loader  # or (lambda _bytes: loads(_bytes))
        self.dumps: Callable = dumper  # or partial(dumps, protocol=5, with_refs=False

        self.queue: deque[bytes] = deque(maxlen=1024 * 1024 * 10)
        self.destinations = []

    def put(self, item: BaseItem) -> None:
        """"""

        # to_js converts objects to JS objects (bytes to Uint8Array)
        if self.destinations:
            destination = self.destinations[0] if len(self.destinations) == 1 else choice(self.destinations)
            destination.postMessage(to_js({"type": self.name, "item": self.dumps(item)}))
        else:
            postMessage(to_js({"type": self.name, "item": self.dumps(item)}))

    def bulk_put(self, items, destination) -> None:
        """"""

        destination.postMessage(to_js({"type": f"{self.name}_bulk", "items": items}))

    def inject(self, raw_item: bytes) -> None:
        """"""

        self.queue.append(raw_item)

    def set_destination(self, port) -> None:
        """"""

        self.destinations = [port]

    def set_mixed_destination(self, ports) -> None:
        """"""

        for port in ports:
            self.destinations.append(port)

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        dumped_items = [self.dumps(item) for item in items]
        if len(self.destinations) == 1:
            self.bulk_put(dumped_items, self.destinations[0])
        else:
            len_items = len(items)
            chunk_size = math.ceil(len_items / len(self.destinations))
            for i, chunk in enumerate(self.chunks(dumped_items, len_items, chunk_size)):
                self.bulk_put(chunk, self.destinations[i])

    @staticmethod
    def chunks(items, len_items, k):
        """Yield successive k-sized chunks from lst."""

        for i in range(0, len_items, k):
            yield items[i:i + k]

    def get(self) -> Optional[BaseItem]:
        """"""

        try:
            return self.loads(self.queue.popleft())
        except IndexError:
            return None

    def get_many(self, max_messages_to_get: int, timeout: float = 0) -> List[BaseItem]:
        """"""

        return [item for item in (self.get() for _ in range(max_messages_to_get)) if item is not None]
        # try:
        #     return [self.loads(item) for item in (self.queue.popleft() for i in range(max_messages_to_get)) if item is not None]
        # except IndexError:
        #     return []

    def _get_many_blocking(
        self, max_messages_to_get: int, timeout: float
    ) -> List[BaseItem]:
        """"""

        return self.get_many(max_messages_to_get, timeout)

    def is_empty(self) -> bool:
        """"""

        return False

    def size(self) -> int:
        """"""

        return self.queue.maxlen

    def close(self) -> None:
        """"""

        self.queue.clear()
