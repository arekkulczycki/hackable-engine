# -*- coding: utf-8 -*-

from typing import Dict, List, Optional

from UltraDict import UltraDict

from arek_chess.common.memory.base_memory import BaseMemory


class UltraDictAdapter(BaseMemory):
    """
    Adapts UltraDict to be used for memory storage.
    """

    def __init__(self):
        self.db: UltraDict = UltraDict(name="hackable-bot")

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self.db[key]
        except KeyError:
            return None

    def get_many(self, keys: List[str]) -> List[bytes]:
        return [self.db[key] for key in keys]

    def set(self, key: str, value: bytes, *, new: bool = True) -> None:
        self.db[key] = value

    def set_many(self, many: Dict[str, bytes]) -> None:
        for key, value in many.items():
            self.db[key] = value

    def remove(self, key: str) -> None:
        del self.db[key]

    def clean(self, except_prefix: str = "", silent: bool = False) -> None:
        self.db.clear()
