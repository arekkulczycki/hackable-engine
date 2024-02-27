# -*- coding: utf-8 -*-
from typing import Callable, Generic, List, Optional, TypeVar

from hackable_engine.common.constants import QUEUE_HANDLER, QueueHandler
from hackable_engine.common.queue.base_queue import BaseQueue

if QUEUE_HANDLER == QueueHandler.FASTER_FIFO:
    from hackable_engine.common.queue.adapters.faster_fifo_adapter import FasterFifoAdapter
elif QUEUE_HANDLER == QueueHandler.REDIS:
    from hackable_engine.common.queue.adapters.redis_adapter import RedisAdapter
elif QUEUE_HANDLER == QueueHandler.RABBITMQ:
    from hackable_engine.common.queue.adapters.rabbitmq_adapter import RabbitmqAdapter
elif QUEUE_HANDLER == QueueHandler.WASM:
    from hackable_engine.common.queue.adapters.wasm_adapter import WasmAdapter

from hackable_engine.common.queue.items.base_item import BaseItem

TItem = TypeVar("TItem", bound=BaseItem)


class QueueManager(Generic[TItem]):
    """"""

    def __init__(
        self,
        name: str,
        loader: Optional[Callable] = None,
        dumper: Optional[Callable] = None,
    ):
        """
        Initialize a queue of a chosen queuing class.
        """

        self.queue: BaseQueue

        if QUEUE_HANDLER == QueueHandler.FASTER_FIFO:
            self.queue = FasterFifoAdapter(name, loader, dumper)
        elif QUEUE_HANDLER == QueueHandler.REDIS:
            self.queue = RedisAdapter(name, loader, dumper)
        elif QUEUE_HANDLER == QueueHandler.RABBITMQ:
            self.queue = RabbitmqAdapter(name)
        elif QUEUE_HANDLER == QueueHandler.WASM:
            self.queue = WasmAdapter(name, loader, dumper)

    @property
    def name(self) -> str:
        return self.queue.name

    def inject(self, raw_item: bytes) -> None:
        """"""

        self.queue.inject(raw_item)

    def inject_js(self, raw_item) -> None:
        """"""

        self.queue.inject(raw_item.to_py().tobytes())

    def set_destination(self, port) -> None:
        """"""

        self.queue.set_destination(port)

    def set_mixed_destination(self, ports) -> None:
        """"""

        self.queue.set_mixed_destination(ports)

    def put(self, item: TItem) -> None:
        """"""

        self.queue.put(item)

    def put_many(self, items: List[TItem]) -> None:
        """"""

        self.queue.put_many(items)

    def get(self) -> Optional[TItem]:
        """"""

        return self.queue.get()

    def get_many(
        self, max_messages_to_get: int = 10, timeout: float = 0.0
    ) -> List[TItem]:
        """"""

        return self.queue.get_many(max_messages_to_get, timeout)

    def get_all(self) -> List[TItem]:
        """"""

        return self.queue.get_all()

    def is_empty(self) -> bool:
        """"""

        return self.queue.is_empty()

    def size(self) -> int:
        """"""

        return self.queue.size()

    def close(self) -> None:
        """"""

        self.queue.close()
