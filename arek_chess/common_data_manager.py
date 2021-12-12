# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import traceback
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict

import numpy
from keydb import KeyDB

SHARED_MEMORY_SIZE = 3


class CommonDataManager:
    """
    Class_docstring
    """

    def __init__(self, memory_manager=None):
        self.db = KeyDB(host="localhost")

        # self.memory_manager = memory_manager

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
        mem = self.get_node_memory(node_name)
        return mem

        # value = self.db.get(node_name)
        # return [float(v) for v in value.split(b"/")]

    def set_params(self, node_name, white_params, black_params):
        self.create_set_node_memory(node_name, [white-black for white, black in zip(white_params, black_params)])

        # concat_params = "/".join(str(white-black) for white, black in zip(white_params, black_params))
        # self.db.set(node_name, concat_params)

    @staticmethod
    def create_node_memory(node_name) -> None:
        size = numpy.dtype(numpy.float16).itemsize * numpy.prod((SHARED_MEMORY_SIZE,))
        SharedMemory(name=node_name, create=True, size=size)

    @staticmethod
    def create_set_node_memory(node_name, param_list) -> None:
        size = numpy.dtype(numpy.float16).itemsize * numpy.prod((SHARED_MEMORY_SIZE,))
        shm = SharedMemory(name=node_name, create=True, size=size)
        data = numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf)
        data[:] = param_list

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
            shm.close()

    @staticmethod
    def remove_node_memory(node_name) -> None:
        try:
            shm = SharedMemory(name=node_name, create=False)
        except FileNotFoundError:
            print(traceback.format_exc())
        else:
            shm.close()
            shm.unlink()

    @staticmethod
    def get_node_memory(node_name) -> Optional[Dict]:
        try:
            shm = SharedMemory(name=node_name)
        except FileNotFoundError:  # TODO: any else errors?
            print(traceback.format_exc())
            return None

        return numpy.ndarray(shape=(SHARED_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf).tolist()
