# -*- coding: utf-8 -*-
"""
All utilities dedicated to manage a shared memory between multiple processes.
"""

import _posixshmem
import mmap
import traceback
from multiprocessing import resource_tracker
from os import O_RDWR, O_EXCL, ftruncate, close, fstat, O_CREAT
from typing import Optional, List

import numpy
from larch import pickle

from arek_chess.board.board import Board

PARAM_MEMORY_SIZE = 5  # TODO: this should be custom for each criteria/evaluator


class MemoryManager:
    """
    Manages the shared memory between multiple processes.
    """

    @classmethod
    def get_node_params(cls, node_name: str) -> List[float]:
        return cls.get_node_memory(f"{node_name}.params")

    @staticmethod
    def get_node_board(node_name: str) -> Optional[Board]:
        shm = DangerousSharedMemory(name=f"{node_name}.board")

        # TODO: tobytes copies the data which could be just read into loads, find improvement
        return pickle.loads(shm.buf.tobytes())

    @staticmethod
    def set_node_board(node_name: str, board: Board) -> None:
        b = pickle.dumps(board, protocol=5, with_refs=False)
        size = len(b)
        try:
            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=size,
            )
        except:
            print(f"board not erased... {node_name}")
            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=False,
                size=size,
            )
            shm.close()
            shm.unlink()
            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=size,
            )

        shm.buf[:] = b

    @staticmethod
    def create_node_memory(node_name: str) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        DangerousSharedMemory(name=node_name, create=True, size=size)

    @staticmethod
    def create_set_node_memory(node_name: str, param_list: List[float]) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        shm = DangerousSharedMemory(name=node_name, create=True, size=size)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = param_list

    @staticmethod
    def set_node_params(node_name: str, *args: float) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        try:
            shm = DangerousSharedMemory(name=f"{node_name}.params", create=True, size=size)
        except:
            print(f"params not erased... {node_name}")
            shm = DangerousSharedMemory(name=f"{node_name}.params", create=False, size=size)
            shm.close()
            shm.unlink()
            shm = DangerousSharedMemory(name=f"{node_name}.params", create=True, size=size)
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = (*args,)

    @staticmethod
    def close_node_memory(node_name: str) -> None:
        try:
            shm = DangerousSharedMemory(name=node_name, create=False)
        except FileNotFoundError:
            print(traceback.format_exc())
        else:
            shm.close()

    @staticmethod
    def remove_node_memory(node_name: str) -> None:
        shm_board = DangerousSharedMemory(name=f"{node_name}.board", create=False)
        shm_params = DangerousSharedMemory(name=f"{node_name}.params", create=False)

        for shm in (shm_board, shm_params):
            shm.close()
            shm.unlink()

    @staticmethod
    def remove_node_params_memory(node_name: str) -> None:
        shm = DangerousSharedMemory(name=f"{node_name}.params", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def remove_node_board_memory(node_name: str) -> None:
        shm = DangerousSharedMemory(name=f"{node_name}.board", create=False)

        shm.close()
        shm.unlink()

    @staticmethod
    def get_node_memory(node_name: str) -> List[float]:
        try:
            shm = DangerousSharedMemory(name=node_name)
        except FileNotFoundError:  # TODO: any else errors?
            print(traceback.format_exc())
            raise

        return numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        ).tolist()


class DangerousSharedMemory:
    """
    A modified copy of multiprocessing SharedMemory

    Named dangerous because I don't know what I'm doing :)
    """

    # Defaults; enables close() and unlink() to run without errors.
    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = O_RDWR
    _mode = 0o600
    _prepend_leading_slash = True

    def __init__(self, name: str = None, create: bool = False, size: int = 0):
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")
        if create:
            _O_CREX = O_CREAT | O_EXCL
            self._flags = _O_CREX | O_RDWR
            if size == 0:
                raise ValueError("'size' must be a positive number different from zero")

        self._fd = _posixshmem.shm_open(name, self._flags, mode=self._mode)
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

        # if create:
        # resource_tracker.register(self._name, "shared_memory")

        self._size = size
        self._buf = memoryview(self._mmap)

    def __del__(self):
        try:
            self.close()
        except OSError:
            pass

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.name,
                False,
                self.size,
            ),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, size={self.size})"

    @property
    def buf(self):
        "A memoryview of contents of the shared memory block."
        return self._buf

    @property
    def name(self):
        "Unique name that identifies the shared memory block."
        reported_name = self._name
        if self._prepend_leading_slash:
            if self._name.startswith("/"):
                reported_name = self._name[1:]
        return reported_name

    @property
    def size(self):
        "Size in bytes."
        return self._size

    def close(self):
        """Closes access to the shared memory from this instance but does
        not destroy the shared memory block."""
        if self._buf is not None:
            self._buf.release()
            self._buf = None
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd >= 0:
            close(self._fd)
            self._fd = -1

    def unlink(self):
        """Requests that the underlying shared memory block be destroyed.

        In order to ensure proper cleanup of resources, unlink should be
        called once (and only once) across all processes which have access
        to the shared memory block."""
        if self._name:
            # resource_tracker.unregister(self._name, "shared_memory")

            _posixshmem.shm_unlink(self._name)


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(resource_tracker._resource_tracker, name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(resource_tracker._resource_tracker, name, rtype)
    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
