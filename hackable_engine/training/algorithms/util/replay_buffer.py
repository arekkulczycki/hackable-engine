# -*- coding: utf-8 -*-
import asyncio
import os.path
import shutil
import traceback
from collections import deque
from enum import Enum
from multiprocessing import Process
from queue import Empty
from random import random
from struct import pack, unpack
from typing import Generator

import numpy as np
import aiofiles
from faster_fifo import Queue, Full

from hackable_engine.common.constants import FLOAT_TYPE

Experience = tuple[np.array, float, float, np.array, bool]


class ReplayBuffer:
    class StorageType(Enum):
        LIST = 0
        DEQUE = 1

    def __init__(self, capacity: int, storage_type=StorageType.DEQUE):
        self.count: int = 0
        self.capacity: int = capacity
        self.storage_type: ReplayBuffer.StorageType = storage_type

        if storage_type is storage_type.DEQUE:
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer = []

        self.backup: DiskBackup | None = None

    def setup_disk_backup(self, capacity: int, batch_size: int, obs_shape: tuple[int, ...], version: int):
        self.in_queue = Queue(maxsize=10 * 1024 * 1024)
        self.out_queue = Queue(maxsize=10 * 1024 * 1024)

        self.backup = DiskBackup(
            self.out_queue, self.in_queue, capacity, batch_size, obs_shape, version
        )
        self.backup.start()

    def push(self, position: int, experience: Experience):
        if self.storage_type is self.StorageType.DEQUE:
            self.buffer.append(experience)
        else:
            if self.count < self.capacity:
                self.count += 1
                self.buffer.append(experience)
            else:
                if self.backup is not None and self.backup.active:
                    # store on disk mostly experiences with non-zero rewards
                    if experience[2] != 0 or random() > 0.95:
                        try:
                            self.out_queue.put_nowait(self.buffer[position])
                        except Full:
                            pass
                self.buffer[position] = experience

    def sample(self, size: int) -> Generator[Experience, None, None]:
        batch_ids = np.random.randint(self.size(), size=size)
        batch = (self.buffer[i] for i in batch_ids)

        if self.backup is not None and self.backup.active and self.size() >= self.capacity:
            try:
                backup_batches = self.in_queue.get_many_nowait(max_messages_to_get=size)
            except Empty:
                pass
            else:
                for i, backup_batch in zip(batch_ids, backup_batches):
                    self.buffer[i] = backup_batch

        return batch

    def size(self):
        if self.storage_type is self.StorageType.DEQUE:
            return len(self.buffer)
        else:
            return self.count


class DiskBackup(Process):
    """
    Constantly store experiences on disk and provide a number (`batch_size`) of experiences in a queue to be consumed.
    """

    def __init__(
        self,
        in_queue: Queue,
        out_queue: Queue,
        size: int,
        batch_size: int,
        obs_shape: tuple[int, ...],
        version: int,
    ):
        super().__init__(daemon=True)

        self.in_queue: Queue = in_queue
        self.out_queue: Queue = out_queue

        self.size = size
        self.filled = False
        self.active = False
        self._write_pointer = 0
        self._read_pointer = 0
        self.batch_size = batch_size
        self.obs_shape = (1, *obs_shape)

        self.keys = ("last_obs", "action", "reward", "obs", "done")
        self.path = f"/var/tmp/hackable_engine/dqn_v{version}_replay_buffer"
        # if os.path.exists(self.path):
        #     shutil.rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)

        self.arr_size = obs_shape[0] * obs_shape[1] * 4

    @property
    def write_pointer(self):
        self._write_pointer += 1
        if self._write_pointer >= self.size:
            self.filled = True
            self._write_pointer = 0
        return self._write_pointer

    @property
    def read_pointer(self):
        self._read_pointer += 1
        if self._read_pointer >= self.size:
            self._read_pointer = 0
        return self._read_pointer

    def run(self):
        asyncio.run(self.work())

    async def work(self):
        while True:
            try:
                experiences = self.in_queue.get_many(block=False)
            except Empty:
                await asyncio.sleep(0.01)
            else:
                coroutines = []
                for experience in experiences:
                    coroutines.append(
                        asyncio.create_task(self.store(self.write_pointer, experience))
                    )
                if coroutines:
                    await asyncio.wait(coroutines)

            if self.out_queue.empty() and (
                self.filled
                or self._write_pointer > self._read_pointer + self.batch_size
            ):
                self.out_queue.put_many(
                    [el for el in await self.prepare_batch() if el is not None]
                )
                await asyncio.sleep(0.01)
            elif self._write_pointer == 0:
                await asyncio.sleep(0.01)

    async def prepare_batch(self):
        return await asyncio.gather(
            *(self.get(self.read_pointer) for _ in range(self.batch_size))
        )

    async def store(self, position: int, experience: Experience):
        value = (
            experience[0].tobytes()
            + pack("f", experience[1])
            + pack("f", experience[2])
            + experience[3].tobytes()
            + (bytes(1) if experience[4] else bytes(0))
        )
        await self.set_file_content(f"replay_buffer_{position}", value)

    async def get(self, position: int) -> Experience | None:

        content = await self.get_file_content(f"replay_buffer_{position}")
        lengths = (self.arr_size, 4, 4, self.arr_size, 1)
        pieces = []
        start = 0
        end = 0
        for l in lengths:
            end += l
            pieces.append(content[start:end])
            start += l

        return (
            np.frombuffer(pieces[0], dtype=FLOAT_TYPE).reshape(self.obs_shape),
            unpack("f", pieces[1])[0],
            unpack("f", pieces[2])[0],
            np.frombuffer(pieces[3], dtype=FLOAT_TYPE).reshape(self.obs_shape),
            bool(pieces[4]),
        )

    async def get_file_content(self, file_name: str) -> bytes:
        async with aiofiles.open(f"{self.path}/{file_name}", mode="rb") as f:
            return await f.read()

    async def set_file_content(self, file_name: str, content: bytes) -> None:
        async with aiofiles.open(f"{self.path}/{file_name}", mode="wb") as f:
            return await f.write(content)
