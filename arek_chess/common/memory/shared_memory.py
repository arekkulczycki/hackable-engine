# -*- coding: utf-8 -*-
"""
Manages the shared memory between multiple processes.
"""

import os
import traceback
from multiprocessing import resource_tracker
from os import O_RDWR, O_EXCL, ftruncate, close, fstat, O_CREAT
from typing import List, Tuple, Optional

import _posixshmem
import mmap
import numpy
from chess import Move
from larch.pickle.pickle import dumps, loads

from arek_chess.board.board import Board
from arek_chess.common.constants import ROOT_NODE_NAME
from arek_chess.common.memory.base_memory import BaseMemory

PARAM_MEMORY_SIZE = 5  # TODO: this should be custom for each criteria/evaluator


class SharedMemory(BaseMemory):
    """
    Manages the shared memory between multiple processes.
    """

    @staticmethod
    def parse_key(key: str) -> str:
        if len(key) > 253:
            return key[-253:].partition(".")[2]
        return key

    def get(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        key = self.parse_key(key)

        try:
            shm = DangerousSharedMemory(name=key)
            return shm.buf.tobytes()
        except:
            if default is not None:
                return default
            # print(f"Error! Not found: {key}")
            # traceback.print_exc()
            return None

    def get_many(self, keys: List[str]) -> List[Optional[bytes]]:
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes) -> None:
        key = self.parse_key(key)

        size = len(value)
        if size == 0:
            raise ValueError(f"Empty value given for: {key}")
        try:
            shm = DangerousSharedMemory(
                name=key,
                create=True,
                size=size,
                write=True,
            )
            shm.buf[:] = value
        except FileExistsError:
            # print(f"Re-created file for: {key}")
            self.remove(key)
            self.set(key, value)
        except:
            print(f"Error! Cannot set: {key}")
            try:
                shm = DangerousSharedMemory(
                    name=key,
                    create=True,
                    size=size,
                    write=True,
                )
                shm.buf[:] = value
                print("second time worked...")
            except:
                traceback.print_exc()
                print(f"Error! Cannot set: {key}")

    def set_many(self, many: List[Tuple[str, bytes]]):
        for key, value in many:
            self.set(key, value)

    def remove(self, key: str) -> None:
        try:
            shm = DangerousSharedMemory(name=key, create=False, write=True)
        except FileNotFoundError:
            pass
        else:
            shm.close()
            shm.unlink()

    def get_action(self):
        return self.get("action")

    @classmethod
    def get_node_params(cls, node_name: str) -> List[float]:
        return cls.get_node_memory(f"{node_name}.params")

    @staticmethod
    def get_node_board(node_name: str) -> Board:
        try:
            shm = DangerousSharedMemory(name=f"{node_name}.board")
        except FileNotFoundError:
            print(f"recreating node {node_name}...")
            return SharedMemory.recreate_node_board(node_name)

        # TODO: tobytes copies the data which could be just read into loads, find improvement
        board = loads(shm.buf.tobytes())
        if not isinstance(board, Board):
            return SharedMemory.recreate_node_board(node_name)
        else:
            return board

    @staticmethod
    def recreate_node_board(node_name: str) -> Board:
        if node_name == ROOT_NODE_NAME:
            raise ValueError(f"cannot recreate root")

        parent_node_name, _, node_move = node_name.rpartition(".")
        board = SharedMemory.get_node_board(parent_node_name)
        board.light_push(Move.from_uci(node_move), state_required=True)

        SharedMemory.set_node_board(node_name, board)

        return board

    @staticmethod
    def set_node_board(node_name: str, board: Board) -> None:
        b = dumps(board, protocol=5, with_refs=False)
        size = len(b)
        try:
            shm = DangerousSharedMemory(
                name=f"{node_name}.board",
                create=True,
                size=size,
            )
        except FileExistsError:
            # FIXME: raises a lot of times
            # print(f"board not erased... {node_name}")
            try:
                shm = DangerousSharedMemory(
                    name=f"{node_name}.board",
                    create=False,
                    size=size,
                )
            except:
                print("I'm sad")
            # shm.close()
            # shm.unlink()
            #
            # shm = DangerousSharedMemory(
            #     name=f"{node_name}.board",
            #     create=True,
            #     size=size,
            # )

        shm.buf[:] = b

    @staticmethod
    def set_node_params(node_name: str, *args: float) -> None:
        size = numpy.dtype(numpy.float16).itemsize * PARAM_MEMORY_SIZE
        try:
            shm = DangerousSharedMemory(
                name=f"{node_name}.params", create=True, size=size
            )
        except FileExistsError:
            # FIXME: raises a lot of times
            # print(f"params not erased... {node_name}")
            shm = DangerousSharedMemory(
                name=f"{node_name}.params", create=False, size=size
            )
            shm.close()
            shm.unlink()

            shm = DangerousSharedMemory(
                name=f"{node_name}.params", create=True, size=size
            )
        data = numpy.ndarray(
            shape=(PARAM_MEMORY_SIZE,), dtype=numpy.float16, buffer=shm.buf
        )
        data[:] = (*args,)

    @staticmethod
    def remove_node_memory(node_name: str) -> None:
        shm_board = DangerousSharedMemory(name=f"{node_name}.board", create=False)
        shm_params = DangerousSharedMemory(name=f"{node_name}.params", create=False)

        for shm in (shm_board, shm_params):
            shm.close()
            shm.unlink()

    @staticmethod
    def remove_node_params_memory(node_name: str) -> None:
        try:
            shm = DangerousSharedMemory(name=f"{node_name}.params", create=False)
        except FileNotFoundError:
            pass
            # print(traceback.format_exc())
        else:
            shm.close()
            shm.unlink()

    @staticmethod
    def remove_node_board_memory(node_name: str) -> None:
        try:
            shm = DangerousSharedMemory(name=f"{node_name}.board", create=False)
        except FileNotFoundError:
            raise
        else:
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

    def clean(self) -> None:
        """"""

        # for filename in os.listdir("/dev/shm"):
        #     path = os.path.join("/dev/shm", filename)
        #     if os.path.isfile(path) and filename.startswith(f"{ROOT_NODE_NAME}."):  # FIXME: incorrect
        #         try:
        #             os.unlink(path)
        #         except FileNotFoundError:
        #             print(f"File {path} not found")
        # try:
        #     os.unlink(os.path.join("/dev/shm", ROOT_NODE_NAME))
        #     os.unlink(os.path.join("/dev/shm", "action"))
        # except:
        #     pass

        # TODO: make this right...
        os.system(f"rm /dev/shm/1.*")
        os.system(f"rm /dev/shm/*")

        # for c in "abcdefgh":
        #     for d in "12345678":
        #         os.system(f"rm /dev/shm/{c}{d}*")

        print("OK")  # just to align with what Redis does :)


class DangerousSharedMemory:
    """
    A modified copy of multiprocessing SharedMemory

    Named "dangerous" because I don't know what I'm doing :)
    """

    # Defaults; enables close() and unlink() to run without errors.
    _name = None
    _fd = -1
    _mmap = None
    _buf = None
    _flags = O_RDWR
    _mode = 0o600
    _prepend_leading_slash = True

    def __init__(self, name: str = None, create: bool = False, size: int = 0, write: bool = False):
        if not size >= 0:
            raise ValueError("'size' must be a positive integer")

        self.do_init(name, create, size, write)

    def do_init(self, name, create, size, write):
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
            access = mmap.ACCESS_WRITE if write else mmap.ACCESS_READ
            self._mmap = mmap.mmap(self._fd, size, access=access)
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
            try:
                _posixshmem.shm_unlink(self._name)
            except:
                pass


def remove_shm_from_resource_tracker():
    """Monkey-patch multiprocessing.resource_tracker so SharedMemory won't be tracked

    More details at: https://bugs.python.org/issue38119
    """

    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(
            resource_tracker._resource_tracker, name, rtype
        )

    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(
            resource_tracker._resource_tracker, name, rtype
        )

    resource_tracker.unregister = fix_unregister

    if "shared_memory" in resource_tracker._CLEANUP_FUNCS:
        del resource_tracker._CLEANUP_FUNCS["shared_memory"]
