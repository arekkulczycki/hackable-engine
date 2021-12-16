# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import _posixshmem
import mmap
import traceback
from io import BytesIO
from multiprocessing.shared_memory import SharedMemory, _make_filename
from os import O_RDWR, O_EXCL, ftruncate, fstat, O_CREAT
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
            shm = DangerousSharedMemory(name=f"{node_name}.board")
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
            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=buffer.itemsize * buffer.nbytes,
            )
        except FileExistsError:
            shm = DangerousSharedMemory(name=f"{node_name}.board", create=False)
            shm.close()
            shm.unlink()

            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=buffer.itemsize * buffer.nbytes,
            )

        shm.buf[:] = buffer

    @staticmethod
    def create_node_memory(node_name) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        DangerousSharedMemory(name=node_name, create=True, size=size)

    @staticmethod
    def create_set_node_memory(node_name, param_list) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        shm = DangerousSharedMemory(name=node_name, create=True, size=size)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = param_list

    @staticmethod
    def set_node_memory(node_name, *args) -> None:
        shm = DangerousSharedMemory(name=node_name, create=False)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = (*args,)

    @staticmethod
    def close_node_memory(node_name) -> None:
        try:
            shm = DangerousSharedMemory(name=node_name, create=False)
        except FileNotFoundError:
            print(traceback.format_exc())
        else:
            shm.close()

    @staticmethod
    def remove_node_memory(node_name) -> None:
        shm_board = DangerousSharedMemory(name=f"{node_name}.board", create=False)
        shm_params = DangerousSharedMemory(name=f"{node_name}.params", create=False)

        for shm in (shm_board, shm_params):
            shm.close()
            shm.unlink()

    @staticmethod
    def remove_node_params_memory(node_name) -> None:
        shm = DangerousSharedMemory(name=f"{node_name}.params", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def remove_node_board_memory(node_name) -> None:
        shm = DangerousSharedMemory(name=f"{node_name}.board", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def get_node_memory(node_name) -> Optional[Dict]:
        try:
            shm = DangerousSharedMemory(name=node_name)
        except FileNotFoundError:  # TODO: any else errors?
            print(traceback.format_exc())
            return None

        return numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        ).tolist()


class DangerousSharedMemory(SharedMemory):
    """
    Named dangerous because I don't know what I'm doing :)
    """

    def __init__(self, name=None, create=False, size=0):
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            _O_CREX = O_CREAT | O_EXCL
            self._flags = _O_CREX | O_RDWR
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")
        if name is None and not self._flags & O_EXCL:
            raise ValueError("'name' can only be None if create=True")

        # POSIX Shared Memory
        if name is None:
            while True:
                name = _make_filename()
                try:
                    self._fd = _posixshmem.shm_open(
                        name,
                        self._flags,
                        mode=self._mode
                    )
                except FileExistsError:
                    continue
                self._name = name
                break
        else:
            name = "/" + name if self._prepend_leading_slash else name
            self._fd = _posixshmem.shm_open(
                name,
                self._flags,
                mode=self._mode
            )
            self._name = name
        try:
            if create and size:
                ftruncate(self._fd, size)
            stats = fstat(self._fd)
            size = stats.st_size
            self._mmap = mmap.mmap(self._fd, size)
        except OSError:
            self.unlink()
            raise

        if create:
            from multiprocessing.resource_tracker import register
            register(self._name, "shared_memory")

        self._size = size
        self._buf = memoryview(self._mmap)
