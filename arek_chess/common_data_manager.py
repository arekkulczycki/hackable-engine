# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import traceback
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict

import numpy
from keydb import KeyDB

SHARED_MEMORY_SIZE = 10
HALF_MEMORY_SIZE = 5


class CommonDataManager:
    """
    Class_docstring
    """

    def __init__(self):
        self.db = KeyDB(host="localhost")

    def get_score(self, key):
        """
        :param key: concatenation of fen and side to play (1 is white)
        """

        return self.db.get(key)

    def set_score(self, key, score):
        """
        :param key: concatenation of fen and side to play (1 is white)
        """

        self.db.set(key, score)

    def get_params(self, node_name):
        value = self.db.get(node_name)
        return [float(v) for v in value.split(b"/")]

    def set_params(self, node_name, white_params, black_params):
        concat_params = "/".join(str(white-black) for white, black in zip(white_params, black_params))
        self.db.set(node_name, concat_params)

    @staticmethod
    def create_node_memory(node_name) -> None:
        size = numpy.dtype(numpy.float16).itemsize * numpy.prod((SHARED_MEMORY_SIZE,))
        shm = SharedMemory(name=node_name, create=True, size=size)
        shm.close()

    @staticmethod
    def create_set_node_memory(node_name, *args) -> None:
        size = numpy.dtype(numpy.float16).itemsize * numpy.prod((SHARED_MEMORY_SIZE,))
        try:
            shm = SharedMemory(name=node_name, create=True, size=size)
        except FileExistsError:
            shm = SharedMemory(name=node_name, create=False, size=size)
        data = numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf)
        data[:] = [round(arg, 2) for arg in args]
        shm.close()

    @staticmethod
    def set_node_memory(node_name, *args) -> None:
        shm = SharedMemory(name=node_name, create=False)
        data = numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf)
        data[:] = (*args,)

    @staticmethod
    def close_node_memory(node_name) -> None:
        try:
            shm = SharedMemory(name=node_name, create=False)
        except FileNotFoundError:
            print(traceback.format_exc())
        else:
            # print(f'unlinking: {node_name}')
            shm.close()
            # shm.unlink()

    @staticmethod
    def remove_node_memory(node_name) -> None:
        try:
            shm = SharedMemory(name=node_name, create=False)
        except FileNotFoundError:
            print(traceback.format_exc())
        else:
            # print(f'unlinking: {node_name}')
            shm.close()
            shm.unlink()

    @staticmethod
    def get_node_memory(node_name) -> Optional[Dict]:
        try:
            shm = SharedMemory(name=node_name, create=False)
        except FileNotFoundError:  # TODO: any else errors?
            return None

        white = numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf)[:HALF_MEMORY_SIZE]
        black = numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf)[HALF_MEMORY_SIZE:]

        return {
            True: dict(zip(("material", "safety", "under_attack", "mobility", "king_mobility"), white)),
            False: dict(zip(("material", "safety", "under_attack", "mobility", "king_mobility"), black)),
        }
