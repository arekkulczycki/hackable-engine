# -*- coding: utf-8 -*-
from collections import deque
from functools import partial
from queue import Empty, Full
from typing import Callable, List, Optional

from js import postMessage, addEventListener
from pyodide.ffi import create_proxy  # to_js
from pyodide.ffi.wrappers import add_event_listener

from arek_chess.common.queue.base_queue import BaseQueue
from arek_chess.common.queue.items.base_item import BaseItem


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
        """

        super().__init__(name)

        self.loads: Callable = loader  # or (lambda _bytes: loads(_bytes))
        self.dumps: Callable = dumper  # or partial(dumps, protocol=5, with_refs=False

        self.queue: deque[BaseItem] = deque(maxlen=1024 * 1024 * 10)

        addEventListener("message", create_proxy(self.on_message))
        # add_event_listener(self, "message", self.on_message)

    @staticmethod
    def memoryview_loader(loader: Callable, memview: memoryview) -> BaseItem:
        item: BaseItem = loader(memview.tobytes())
        return item

    @staticmethod
    def memoryview_dumper(dumper: Callable, memview: memoryview) -> Callable:
        return dumper(memview.tobytes())

    def put(self, item: BaseItem) -> None:
        """"""
        print("posting to js")
        # postMessage(to_js(self.dumps(item)))  # to_js converts objects to JS objects
        postMessage(self.dumps(item))

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        for item in items:
            self.put(item)

    def on_message(self, event):
        print("received in py worker: ", event)
        self.queue.append(self.loads(event.data))

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
