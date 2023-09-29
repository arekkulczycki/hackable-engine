# -*- coding: utf-8 -*-
import os
import string
import traceback
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from os import O_CREAT, O_EXCL, O_RDWR, close, fstat, ftruncate
from typing import List, Optional, Tuple

import _posixshmem
import mmap

from arek_chess.common.constants import ACTION, DISTRIBUTED, STATUS
from arek_chess.common.memory.base_memory import BaseMemory

PARAM_MEMORY_SIZE = 5  # TODO: this should be custom for each criteria/evaluator
MAX_LENGTH = 253

uci_chars = "12345678abcdefghnrqk."
"""All characters used in the game notation, used for node identification."""

all_chars = string.digits + string.ascii_lowercase + string.ascii_uppercase + "-_."
"""All characters safe to use as file name."""


def _convert(filename, from_chars, to_chars):
    x = 0
    for digit in str(filename):
        try:
            x = x * len(from_chars) + from_chars.index(digit)
        except ValueError:
            raise ValueError('invalid digit "%s"' % digit)

    # create the result in base 'len(to_digits)'
    if x == 0:
        res = to_chars[0]
    else:
        res = ""
        k = 0
        while x > 0:
            digit = x % len(to_chars)
            res = to_chars[digit] + res
            x = int(x // len(to_chars))
            k += 1
            if k > 100:
                print("fuck it's this")
                print(filename)
    return res


class SharedMemoryAdapter(BaseMemory):
    """
    Manages the shared memory between multiple processes.
    """

    def __init__(self):
        """"""

        # remove_shm_from_resource_tracker()

    @staticmethod
    def parse_key(key: str) -> str:
        # FIXME: distributor will raise on not finding parent, this is temporary solution
        #  or is it? it should allow for depth up to ~35 moves!
        if len(key) > MAX_LENGTH:
            key = _convert(key, uci_chars, all_chars)
        if len(key) > MAX_LENGTH:
            return key[-MAX_LENGTH:].partition(".")[2]
        return key

    def get(self, key: str, default: Optional[bytes] = None) -> Optional[bytes]:
        key = self.parse_key(key)

        try:
            shm = DangerousSharedMemory(name=key)
            # shm = SharedMemory(name=key)
            return shm.buf.tobytes()
        except:
            if default is not None:
                return default
            # print(f"Error! Not found: {key}")
            # traceback.print_exc()
            return None

    def get_many(self, keys: List[str]) -> List[Optional[bytes]]:
        return [self.get(key) for key in keys]

    def set(self, key: str, value: bytes, *, new: bool = True) -> None:
        key = self.parse_key(key)
        # print(key)

        size = len(value)
        if size == 0:
            raise ValueError(f"Empty value given for: {key}")
        try:
            shm = DangerousSharedMemory(
            # shm = SharedMemory(
                name=key,
                create=new,
                size=size,
                write=True,
            )
            shm.buf[:] = value
        except FileExistsError:
            # print(f"Re-created file for: {key}")
            self.remove(key)
            self.set(key, value)
        except FileNotFoundError:
            self.set(key, value)
        except Exception as e:
            print(f"Error! Cannot set: {key}. {e}")
            try:
                shm = DangerousSharedMemory(
                # shm = SharedMemory(
                    name=key,
                    create=new,
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
            # shm = SharedMemory(name=key, create=False)
        except FileNotFoundError:
            pass
        else:
            shm.close()
            shm.unlink()

    def get_action(self):
        return self.get("action")

    def clean(self, except_prefix: str = "", silent: bool = False) -> None:
        """"""

        if len(except_prefix) > MAX_LENGTH:
            if not silent:
                print("OK")
            return  # FIXME: change node identification method

        for filename in os.listdir(
            "/dev/shm"
        ):  # FIXME: dangerous, may remove files not owned by us
            path = os.path.join("/dev/shm", filename)
            parent_name = ".".join(except_prefix.split(".")[:-1])
            if (
                os.path.isfile(path)
                and not filename == parent_name
                and not (except_prefix and filename.startswith(except_prefix))
            ):
                try:
                    os.unlink(path)
                except (FileNotFoundError, PermissionError):
                    # print(f"File {path} not found")
                    pass
        try:
            os.unlink(os.path.join("/dev/shm", ACTION))
            os.unlink(os.path.join("/dev/shm", DISTRIBUTED))
            os.unlink(os.path.join("/dev/shm", STATUS))
        except:
            pass

        # for c in "abcdefgh":
        #     for d in "12345678":
        #         os.system(f"rm /dev/shm/{c}{d}*")

        if not silent:
            if except_prefix:
                print(f"OK (skipped {except_prefix}*)")
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

    def __init__(
        self, name: str = None, create: bool = False, size: int = 0, write: bool = False
    ):
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
