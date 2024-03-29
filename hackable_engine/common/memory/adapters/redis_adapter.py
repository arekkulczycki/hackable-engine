# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional

from redis import Redis
from redis.asyncio import Redis as AsyncRedis

from hackable_engine.common.memory.base_memory import BaseMemory


class RedisAdapter(BaseMemory):
    """
    Adapts Redis to be used for memory storage.
    """

    def __init__(self):
        self.db: Redis = Redis()
        self.async_db: AsyncRedis = AsyncRedis()

    def get(self, key: str) -> Optional[bytes]:
        return self.db.get(key)

    def get_many(self, keys: List[str]):
        return self.db.mget(keys)

    def set(self, key: str, value: bytes, *, new: bool = True):
        self.db.set(key, value)

    def set_many(self, many: Dict[str, bytes]):
        self.db.mset(many)

    def remove(self, key: str):
        self.db.delete(key)

    def get_action(self):
        return self.db.get("action")

    def set_action(self, action) -> None:
        self.db.set("action", action)

    def clean(self, *args, **kwargs) -> None:
        os.system("redis-cli FLUSHALL")

    async def get_async(self, key: str) -> bytes:
        return await self.async_db.get(key)

    async def get_many_async(self, keys: List[str]):
        return await self.async_db.mget(keys)

    async def set_async(self, key: str, value: bytes):
        await self.async_db.set(key, value)

    async def set_many_async(self, many: Dict[str, bytes]):
        await self.async_db.mset(many)

    async def remove_async(self, key: str):
        await self.async_db.delete(key)
