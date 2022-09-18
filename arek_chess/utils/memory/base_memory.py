# -*- coding: utf-8 -*-
"""
Base class to manage memory between multiple processes.
"""

from typing import Dict, List


class BaseMemory:
    """
    Base class to manage memory between multiple processes.
    """

    def get(self, key: str) -> bytes:
        raise NotImplementedError

    def get_many(self, keys: List[str]):
        raise NotImplementedError

    def set(self, key: str, value: bytes):
        raise NotImplementedError

    def set_many(self, many: Dict[str, bytes]):
        raise NotImplementedError

    def remove(self, key: str):
        raise NotImplementedError
