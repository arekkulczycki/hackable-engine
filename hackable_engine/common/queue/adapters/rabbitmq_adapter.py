# -*- coding: utf-8 -*-

from typing import List, Optional, Generator

import pika
from larch.pickle.pickle import dumps, loads

from hackable_engine.common.queue.base_queue import BaseQueue
from hackable_engine.common.queue.items.base_item import BaseItem


class RabbitmqAdapter(BaseQueue):
    """
    Queue provided by external RabbitMQ service.

    FIXME: doesn't work at this moment
    """

    def __init__(self, name: str):
        super().__init__(name)

        self._setup_connection()

        self.channel.queue_declare(queue=name, durable=True)
        self.channel.queue_bind(name, "arek-chess", name)

    def _setup_connection(self) -> None:
        """"""

        url = "amqp://guest:guest@localhost/%2f"
        params = pika.URLParameters(url)
        params.socket_timeout = 5

        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.exchange_declare("arek-chess")

    def put(self, item: BaseItem) -> None:
        """"""

        self.channel.basic_publish("arek-chess", self.name, dumps(item, protocol=5, with_refs=False))

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        for item in items:
            self.put(item)

    def _get(self, timeout: float = 0) -> Generator[bytes, None, None]:
        """"""

        return (response[2] for response in self.channel.consume(self.name, auto_ack=True, inactivity_timeout=timeout))

    def get(self, timeout: float = 0) -> Optional[BaseItem]:
        """"""

        item: bytes = next(self._get(timeout))
        return item and loads(item)

    def get_many(self, max_messages_to_get: int, timeout: float = 0) -> List[BaseItem]:
        """"""

        timeout_ = timeout / max_messages_to_get

        items: List[BaseItem] = []
        for _ in range(max_messages_to_get):
            item = self.get(timeout_)
            if item:
                items.append(item)

        return items

    def is_empty(self) -> bool:
        """"""

        return False

    def size(self) -> int:
        """"""

        return 0

    def close(self) -> None:
        """"""

        self.connection.close()
