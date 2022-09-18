# -*- coding: utf-8 -*-
"""
Manages the memory between multiple processes by storing in Redis.
"""

from typing import Dict, List

from redis import Redis, ConnectionPool

from arek_chess.utils.memory.base_memory import BaseMemory


class RedisMemory(BaseMemory):
    """
    Manages the memory between multiple processes by storing in Redis.
    """

    def __init__(self):
        self.db: Redis = Redis()

    def get(self, key: str) -> bytes:
        return self.db.get(key)

    def get_many(self, keys: List[str]):
        return self.db.mget(keys)

    def set(self, key: str, value: bytes):
        self.db.set(key, value)

    def set_many(self, many: Dict[str, bytes]):
        self.db.mset(many)

    def remove(self, key: str):
        self.db.delete(key)
