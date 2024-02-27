# -*- coding: utf-8 -*-
from typing import Dict, List, Optional

from pyodide.ffi import JsProxy

from hackable_engine.common.memory.base_memory import BaseMemory

KEY_MAP = {
    "worker_0": (0, 1),
    "worker_1": (1, 2),
    "worker_2": (2, 3),
    "worker_3": (3, 4),
    "worker_4": (4, 5),
    "worker_5": (5, 6),
    "worker_6": (6, 7),
    "worker_7": (7, 8),
    "worker_8": (8, 9),
    "worker_9": (9, 10),
    "status": (10, 14),
    "debug": (14, 15),
    "distributed": (15, 19),
    "run_id": (19, 26),
}


class WasmAdapter(BaseMemory):
    """
    Adapts WebAssembly.Memory to be used for memory storage.

    Memory object is received by worker in a message from main thread.
    """

    def __init__(self, memory):
        self.set_memory(memory)

    def set_memory(self, memory) -> None:
        self.db: JsProxy = memory  # JsProxy to memoryview

    def get(self, key: str) -> Optional[bytes]:
        idx = KEY_MAP[key]
        try:
            return self.db.to_py()[idx[0]:idx[1]].tobytes()
        except IndexError:
            return None

    def get_many(self, keys: List[str]) -> List[bytes]:
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes, *, new: bool = True) -> None:
        # self.db[self.key_to_index(key)] = value
        idx = KEY_MAP[key][0]
        for i, b in enumerate(value):
            self.db[i + idx] = b

    def set_many(self, many: Dict[str, bytes]) -> None:
        for key, value in many.items():
            self.set(key, value)

    def remove(self, key: str) -> None:
        del self.db[self.key_to_index(key)]  # TODO: implement

    @staticmethod
    def key_to_index(key):
        try:
            return int(key)  # pack("i", key)
        except ValueError:
            return KEY_MAP[key]

    def clean(self, except_prefix: str = "", silent: bool = False) -> None:
        self.db.clear()
