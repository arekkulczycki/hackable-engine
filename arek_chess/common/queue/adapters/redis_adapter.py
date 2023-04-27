# -*- coding: utf-8 -*-

from typing import List, Optional

from larch.pickle.pickle import dumps, loads
from redis import Redis
from redis.exceptions import ResponseError

from arek_chess.common.queue.base_queue import BaseQueue
from arek_chess.common.queue.items.base_item import BaseItem


class RedisAdapter(BaseQueue):
    """
    Queue provided by external Redis service.
    """

    def __init__(self, name: str):
        super().__init__(name)

        self.broker = Redis("localhost", db=1)
        self.result_backend = Redis("localhost", db=2)

    def put(self, item: BaseItem) -> None:
        """"""

        self.broker.rpush(self.name, dumps(item, protocol=5, with_refs=False))

    def put_many(self, items: List[BaseItem]) -> None:
        """"""

        # for item in items:
        #     self.put(item)
        if not items:
            return

        try:
            self.broker.rpush(self.name, *(dumps(item, protocol=5, with_refs=False) for item in items))
        except ResponseError:
            print(f"too many args? {len(items)}")

    def get(self, timeout: float = 0) -> Optional[BaseItem]:
        """"""

        item: bytes = self.broker.lpop(self.name)
        return item and loads(item)

    def get_many(self, max_messages_to_get: int, timeout: float = 0) -> List[BaseItem]:
        """"""

        # timeout_ = timeout / max_messages_to_get

        # k: int = 1
        # items: List[BaseItem] = []
        # item = self.get()
        # while item and k < max_messages_to_get:
        #     items.append(item)
        #     item = self.get()
        #     k += 1
        #
        # return items

        # TODO: how about BLPOP?
        byte_items: List[bytes] = self.broker.lpop(self.name, max_messages_to_get)
        return [loads(item) for item in byte_items if item] if byte_items else []

    def is_empty(self) -> bool:
        """"""

        return False

    def size(self) -> int:
        """"""

        return 0

    def close(self) -> None:
        """"""
