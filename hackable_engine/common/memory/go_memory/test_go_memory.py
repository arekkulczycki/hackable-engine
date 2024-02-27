# -*- coding: utf-8 -*-

import ctypes
import os

import larch
from chess import Board
from larch import pickle

from hackable_engine.common.memory_manager import DangerousSharedMemory

path = os.path.join(os.path.abspath(__file__).replace(os.path.basename(__file__), ''), "memory.so")
memory = ctypes.CDLL(path)

memory.create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p]
memory.create.restype = None

memory.get.argtypes = [ctypes.c_char_p]
memory.get.restype = ctypes.c_char_p

test_str = "Ala ma kota!".encode("utf-8")
tag = "test".encode("utf-8")
shm_name = "non-existing-name-9"


def test():
    memory.create(tag, 12, test_str)
    value = memory.get(tag)

    assert value == test_str

    test_value = pickle.dumps(Board(), protocol=5, with_refs=False)
    size = len(test_value)

    memory.create("other".encode(), size, )
    value = memory.get("other".encode())

    print(value, test_value)

    assert value == test_value


def benchmark():
    from time import time

    test_value = pickle.dumps(Board(), protocol=5, with_refs=False)
    size = len(test_value)

    # memory.create(tag, size, test_str)

    t0 = time()
    for i in range(10000):
        memory.create(tag, size, test_value)
        value = memory.get(tag)
    print(f"go time: {time() - t0}")

    # shm = DangerousSharedMemory(
    #     name=shm_name,
    #     create=True,
    #     size=size,
    # )
    # shm.buf[:] = test_value
    # shm.close()

    t0 = time()
    for i in range(10000):
        shm = DangerousSharedMemory(
            name=shm_name,
            create=True,
            size=size,
        )
        shm.buf[:] = test_value
        shm.close()
        shm = DangerousSharedMemory(
            name=shm_name,
            create=False,
            size=size,
        )
        value = shm.buf.tobytes()
        shm.close()
        shm.unlink()
    print(f"py time: {time() - t0}")

    # shm.close()
    # shm.unlink()


test()
# benchmark()
