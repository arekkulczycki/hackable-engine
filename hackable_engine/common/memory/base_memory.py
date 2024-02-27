# -*- coding: utf-8 -*-

from typing import Dict, List, Tuple, Optional


class BaseMemory:
    """
    Base class to manage memory between multiple processes.
    """

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def get_many(self, keys: List[str]) -> List[Optional[bytes]]:
        raise NotImplementedError

    def set(self, key: str, value: bytes, *, new: bool = True):
        raise NotImplementedError

    def set_many(self, many: List[Tuple[str, bytes]]):
        raise NotImplementedError

    def remove(self, key: str):
        raise NotImplementedError

    def clean(self, except_prefix: str = ""):
        raise NotImplementedError

    async def get_async(self, key: str) -> bytes:
        raise NotImplementedError

    async def get_many_async(self, keys: List[str]):
        raise NotImplementedError

    async def set_async(self, key: str, value: bytes):
        raise NotImplementedError

    async def set_many_async(self, many: Dict[str, bytes]):
        raise NotImplementedError

    async def remove_async(self, key: str):
        raise NotImplementedError
