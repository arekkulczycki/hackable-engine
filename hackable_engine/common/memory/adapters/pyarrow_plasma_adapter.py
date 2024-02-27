# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

import pyarrow.plasma as plasma


class PyarrowPlasmaMemory:
    """
    Manages the shared memory between multiple processes.
    """

    def __init__(self):
        self.client = plasma.connect("/tmp/plasma")

    def get(self, key: str) -> bytes:
        return self.client.get(plasma.ObjectID(key.rjust(20, "=").encode()))

    def get_many(self, keys: List[str]) -> List[bytes]:
        return self.client.get([plasma.ObjectID(key.rjust(20, "=").encode()) for key in keys])

    def set(self, key: str, value: bytes) -> None:
        obj = plasma.ObjectID(key.rjust(20, "=").encode())
        self.client.create_and_seal(obj, value)

    def set_many(self, many: List[Tuple[str, bytes]]) -> None:
        for key, value in many:
            self.set(key, value)

    def clean(self) -> None:
        """"""

        for filename in os.listdir("/dev/shm"):
            path = os.path.join("/dev/shm", filename)
            if os.path.isfile(path) and filename.startswith("0."):
                os.unlink(path)

        print("OK")  # just to align with what Redis does :)
