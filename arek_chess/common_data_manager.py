# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import traceback
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Dict

import numpy
from larch import pickle

PARAM_MEMORY_SIZE = 5


class CommonDataManager:
    """
    Class_docstring
    """

    @classmethod
    def get_node_params(cls, node_name):
        return cls.get_node_memory(f"{node_name}.params")

    @classmethod
    def set_node_params(cls, node_name, white_params, black_params):
        cls.create_set_node_memory(
            f"{node_name}.params",
            [white - black for white, black in zip(white_params, black_params)],
        )

    @staticmethod
    def get_node_board(node_name):
        try:
            shm = SharedMemory(name=f"{node_name}.board")
        except FileNotFoundError:  # TODO: any else errors?
            print(traceback.format_exc())
            return None

        # TODO: tobytes copies the data which could be just read into loads, find improvement
        return pickle.loads(shm.buf.tobytes())

    @staticmethod
    def set_node_board(node_name, board):
        stream = BytesIO()
        pickle.dump(board, stream, protocol=5)
        buffer = stream.getbuffer()
        try:
            shm = SharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=buffer.itemsize * buffer.nbytes,
            )
        except FileExistsError:
            shm = SharedMemory(name=f"{node_name}.board", create=False)
            shm.close()
            shm.unlink()

            shm = SharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=buffer.itemsize * buffer.nbytes,
            )

        shm.buf[:] = buffer

    @staticmethod
    def create_node_memory(node_name) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        SharedMemory(name=node_name, create=True, size=size)

    @staticmethod
    def create_set_node_memory(node_name, param_list) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        shm = SharedMemory(name=node_name, create=True, size=size)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = param_list

    @staticmethod
    def set_node_memory(node_name, *args) -> None:
        shm = SharedMemory(name=node_name, create=False)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
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
        shm_board = SharedMemory(name=f"{node_name}.board", create=False)
        shm_params = SharedMemory(name=f"{node_name}.params", create=False)

        for shm in (shm_board, shm_params):
            shm.close()
            shm.unlink()

    @staticmethod
    def remove_node_params_memory(node_name) -> None:
        shm = SharedMemory(name=f"{node_name}.params", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def remove_node_board_memory(node_name) -> None:
        shm = SharedMemory(name=f"{node_name}.board", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def get_node_memory(node_name) -> Optional[Dict]:
        try:
            shm = SharedMemory(name=node_name)
        except FileNotFoundError:  # TODO: any else errors?
            print(traceback.format_exc())
            return None

        return numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        ).tolist()
