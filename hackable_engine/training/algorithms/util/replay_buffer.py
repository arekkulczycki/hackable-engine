import asyncio
import os.path
from collections import deque, defaultdict
from enum import Enum
from multiprocessing import Process
from queue import Empty
from struct import pack, unpack
from typing import Generator

import aiofiles
import numpy as np
from faster_fifo import Queue, Full

from hackable_engine.board.hex.bitboard_utils import (
    generate_masks,
    split_mask_to_uint64_array,
)
from hackable_engine.common.constants import FLOAT_TYPE

Experience = tuple[np.ndarray, float, float, np.ndarray, bool, int, int]
IndexedExperience = tuple[np.ndarray, float, float, np.ndarray, bool, int, int, int]


class ReplayBuffer:
    class StorageType(Enum):
        LIST = 0
        DEQUE = 1

    def __init__(
        self,
        board_size: int,
        capacity: int,
        storage_type=StorageType.DEQUE,
        priority_rate: float = 0.0,
        control_stats: bool = True,
    ):
        self.count: int = 0
        self.board_size: int = board_size
        self.board_size_squared: int = board_size**2
        self.board_mask: int = (1 << self.board_size_squared) - 1
        self._capacity: int
        self.capacity: int = capacity
        self.td_errors: np.ndarray = np.zeros(capacity, dtype=FLOAT_TYPE)
        self.storage_type: ReplayBuffer.StorageType = storage_type

        if storage_type is storage_type.DEQUE:
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer = []

        self.win_indices: set[int] = set()
        self.loss_indices: set[int] = set()
        self.illegal_indices: set[int] = set()
        self.priority_rate: float = priority_rate
        """Between 0 and 1, the % of sampled items took from the priority pool."""

        self.control_stats_dict: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
        self.control_stats_keys: dict[int, set[tuple[int, int]]] = defaultdict(set)
        self.control_stats_terminal_keys: set[tuple[int, int]] = set()
        # self.control_stats_keys_arr: np.array = np.empty(shape=(2_000_000, 2, 3), dtype=np.uint64)  # 91 MB
        """Keep track of what keys are already present for the data and facilitate quick subset lookups."""

        self.backup: DiskBackupProcess | None = None
        self.control_stats: bool = control_stats
        self.should_sample_illegal: bool = True

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value: int):
        self._capacity = value
        self.td_errors = np.zeros(value, dtype=FLOAT_TYPE)

    def setup_disk_backup(self, capacity: int, batch_size: int, obs_shape: tuple[int, ...], version: int):
        self.in_queue = Queue(maxsize=10 * 1024 * 1024)
        self.out_queue = Queue(maxsize=10 * 1024 * 1024)

        self.backup = DiskBackupProcess(self.out_queue, self.in_queue, capacity, batch_size, obs_shape, version)
        self.backup.start()

    def push(self, position: int, experience: Experience):
        if self.storage_type is self.StorageType.DEQUE:
            self.buffer.append(experience)
        else:
            if self.count < self.capacity:
                self.buffer.append(experience)
                position = self.count
                self.count += 1
            else:
                if self.backup is not None and self.backup.active:
                    # store on disk mostly experiences with non-zero rewards
                    if experience[2] != 0:  # or random() > 0.95:
                        try:
                            self.out_queue.put_nowait(self.buffer[position])
                        except Full:
                            pass
                self.buffer[position] = experience

            if self.control_stats:
                self.collect_control_stats(experience, position)

    def collect_control_stats(self, experience: Experience, position: int):
        reward = experience[2]
        is_terminal = reward != 0
        is_win = reward > 0
        is_legal = reward >= -1
        ocb = experience[-2]
        ocw = experience[-1]
        if is_legal and ocb not in self.control_stats_dict or ocw not in self.control_stats_dict[ocb]:
            self.control_stats_dict[ocb][ocw] = self.initialize_oc_stats(ocb, ocw)
            # self.control_stats_keys_arr[position][0] = split_mask_to_uint64_array(ocb)
            # self.control_stats_keys_arr[position][1] = split_mask_to_uint64_array(ocw)
            if not is_terminal:
                self.control_stats_keys[ocb.bit_count()].add((ocb, ocw))

        if is_terminal and is_legal:  # prioritize non-zero rewards
            if is_win:
                self.win_indices.add(position)
                self.loss_indices.discard(position)
                self.illegal_indices.discard(position)
            elif is_legal:
                self.loss_indices.add(position)
                self.win_indices.discard(position)
                self.illegal_indices.discard(position)
            elif self.should_sample_illegal:
                self.illegal_indices.add(position)
                self.loss_indices.discard(position)
                self.win_indices.discard(position)

            if is_legal:
                self.propagate_oc_to_children(ocb, ocw)
                self.control_stats_terminal_keys.add((ocb, ocw))
        else:
            for set_ in [self.win_indices, self.loss_indices, self.illegal_indices]:
                set_.discard(position)

    def propagate_oc_to_children(self, ocb: int, ocw: int):
        """Find all keys that have a subset of white stones AND a subset of black stones."""

        # for ocb_child, ocw_child in self.find_mask_subsets_vectorized(ocb, ocw):
        #     child_control_stats = self.control_stats_dict[ocb_child][ocw_child]
        #     # iterate over all cells in the board
        #     for mask in generate_masks(self.board_mask):
        #         c = mask.bit_length() - 1
        #         if mask & ocb:
        #             child_control_stats[c][0] += 1
        #         elif mask & ocw:
        #             child_control_stats[c][1] += 1
        #         else:
        #             child_control_stats[c][2] += 1

        max_count = ocb.bit_count() - 1
        for count in range(max_count):
            for ocb_child, ocw_child in self.control_stats_keys[count]:
                if ocb_child & ocb != ocb_child or ocw_child & ocw != ocw_child:
                    continue

                child_control_stats = self.control_stats_dict[ocb_child][ocw_child]
                self.increment_oc_stats(child_control_stats, ocb, ocw)

    def increment_oc_stats(self, arr: np.ndarray, ocb: int, ocw: int) -> np.ndarray:
        # iterate over all cells in the board
        for mask in generate_masks(self.board_mask):
            c = mask.bit_length() - 1
            if mask & ocb:
                arr[c][0] += 1
            elif mask & ocw:
                arr[c][1] += 1
            else:
                arr[c][2] += 1

    def initialize_oc_stats(self, ocb: int, ocw: int) -> np.ndarray:
        arr = np.zeros(dtype=FLOAT_TYPE, shape=(self.board_size_squared, 3))

        # increment with itself to avoid zeros
        self.increment_oc_stats(arr, ocb, ocw)

        # add from all terminal experiences where current is subset of ocb and ocw
        # TODO: check performance, is this worth it?
        for ocb_parent, ocw_parent in self.control_stats_terminal_keys:
            if ocb & ocb_parent != ocb or ocw & ocw_parent != ocw:
                continue
            self.increment_oc_stats(arr, ocb_parent, ocw_parent)

        return arr

    def find_closest_prob_oc_stats(self, ocb: int, ocw: int):
        if ocb in self.control_stats_dict and ocw in self.control_stats_dict[ocb]:
            return self.get_prob_oc_stats(self.control_stats_dict[ocb][ocw])

        keys = self.control_stats_keys[ocb.bit_count() - 1]
        for ocb_child, ocw_child in keys:
            # check if it's a subset of both ocb and ocw
            if ocb_child & ocb != ocb_child or ocw_child & ocw != ocw_child:
                continue

            return self.get_prob_oc_stats(self.control_stats_dict[ocb_child][ocw_child])
        return self.get_avg_prob_oc_stats()

    @staticmethod
    def get_prob_oc_stats(oc_stats: np.ndarray) -> np.ndarray:
        return oc_stats / np.sum(oc_stats, axis=-1, keepdims=True)

    def get_avg_prob_oc_stats(self) -> np.ndarray:
        return np.zeros(dtype=FLOAT_TYPE, shape=(self.board_size_squared, 3)) + 1 / 3

    def find_mask_subsets_vectorized(self, ocb: int, ocw: int) -> Generator[tuple[int, int], None, None]:
        # Convert target into 3-part chunks
        ocb_chunks = split_mask_to_uint64_array(ocb)
        ocw_chunks = split_mask_to_uint64_array(ocw)

        child_ocb_arr = self.control_stats_keys_arr[:, 0, :]  # shape (N, 3)
        child_ocw_arr = self.control_stats_keys_arr[:, 1, :]  # shape (N, 3)

        # Check subset condition: (x & target) == x for all 3 chunks
        condition_ocb = np.all((child_ocb_arr & ocb_chunks) == child_ocb_arr, axis=1)
        condition_ocw = np.all((child_ocw_arr & ocw_chunks) == child_ocw_arr, axis=1)

        subset_arr = self.control_stats_keys_arr[condition_ocb & condition_ocw]  # shape (N, 2, 3)
        return self.subset_arr_to_pairs(subset_arr)
        # Combine masks
        # mask = mask_A & mask_B
        # return [self.control_stats_keys_arr[i] for i in np.where(condition_ocb & condition_ocw)[0]]

    def subset_arr_to_pairs(self, subset_arr) -> Generator[tuple[int, int], None, None]:
        yield from (
            (
                self.combine_chunks_to_int(pair[0]),  # first bitmask
                self.combine_chunks_to_int(pair[1]),  # second bitmask
            )
            for pair in subset_arr
        )

    @staticmethod
    def combine_chunks_to_int(chunks):
        return int(chunks[0]) + (int(chunks[1]) << 64) + (int(chunks[2]) << 128)

    def sample(self, size: int) -> Generator[IndexedExperience, None, None]:
        priority_size = int(size * self.priority_rate)
        if self.should_sample_illegal:
            win_size = min(len(self.win_indices), priority_size // 3)
            loss_size = min(len(self.loss_indices), (priority_size - win_size) // 2)
            illegal_size = min(len(self.illegal_indices), priority_size - win_size - loss_size)
        else:
            win_size = min(len(self.win_indices), priority_size // 2)
            loss_size = min(len(self.loss_indices), priority_size - win_size)
            illegal_size = 0
        priority_size = win_size + loss_size
        priority_batch_ids = np.concat(
            [
                np.random.choice(list(self.win_indices), size=win_size, replace=False),
                np.random.choice(list(self.loss_indices), size=loss_size, replace=False),
                np.random.choice(list(self.illegal_indices), size=illegal_size, replace=False),
            ]
        )

        if self.size() == self.capacity:
            td_errors = self.td_errors
            weights_sum = self.td_errors.sum()
        else:
            td_errors = self.td_errors[:self.size()]
            weights_sum = td_errors.sum()
        if weights_sum > 0.0:
            batch_ids = np.random.choice(
                self.size(), size=(size - priority_size), replace=False, p=td_errors / weights_sum
            )
        else:
            batch_ids = np.random.choice(self.size(), size=(size - priority_size), replace=False)
        batch = ((*self.buffer[i], i) for i in np.concat((batch_ids, priority_batch_ids)).astype(int))

        if self.backup is not None and self.backup.active and self.size() >= self.capacity:
            try:
                backup_batches = self.in_queue.get_many_nowait(max_messages_to_get=size)
            except Empty:
                pass
            else:
                for i, backup_batch in zip(batch_ids, backup_batches):
                    self.buffer[i] = backup_batch

        return batch

    def size(self) -> int:
        if self.storage_type is self.StorageType.DEQUE:
            return len(self.buffer)
        else:
            return self.count

    async def dump_to_disk(self, obs_shape: tuple[int, ...], path: str | None = None):
        # disk = DiskBackup(obs_shape, "/var/tmp/hackable_engine/initial_buffer")
        # disk = DiskBackup(obs_shape, "/var/tmp/hackable_engine/initial_buffer_legal")
        disk = DiskBackup(obs_shape, path or "/var/tmp/hackable_engine/init_buffer_heuristic")
        for position, experience in enumerate(self.buffer):
            await disk.store(position, experience)

    async def load_init_buffer(self, obs_shape: tuple[int, ...], size: int, path: str | None = None):
        # disk = DiskBackup(obs_shape, "/var/tmp/hackable_engine/initial_buffer")
        # disk = DiskBackup(obs_shape, "/var/tmp/hackable_engine/initial_buffer_legal")
        disk = DiskBackup(obs_shape, path or "/var/tmp/hackable_engine/init_buffer_heuristic")
        # disk = DiskBackup(obs_shape, path or "/var/tmp/hackable_engine/final_buffer")

        for position in range(size):
            experience = await disk.get(position)
            if experience:
                self.push(position, experience)
            else:
                break


class DiskBackup:

    def __init__(self, obs_shape: tuple[int, ...], path: str | None = None):
        self.keys = ("last_obs", "action", "reward", "obs", "done")
        self.path = path or "/var/tmp/hackable_engine/replay_buffer"
        # if os.path.exists(self.path):
        #     shutil.rmtree(self.path)
        os.makedirs(self.path, exist_ok=True)

        self.obs_shape = (1, *obs_shape)
        self.arr_size = obs_shape[0] * obs_shape[1] * 4  # float32 is 4 bytes

    async def store(self, position: int, experience: Experience):
        value = (
            experience[0].tobytes()
            + pack("f", experience[1])
            + pack("f", experience[2])
            + experience[3].tobytes()
            + (bytes([1 if experience[4] else 0]))
            + experience[5].to_bytes(22, byteorder="big", signed=False)  # TODO: 22 depends on board size, here 13x13
            + experience[6].to_bytes(22, byteorder="big", signed=False)
        )
        await self.set_file_content(f"replay_buffer_{position}", value)

    async def get(self, position: int) -> Experience | None:
        try:
            content = await self.get_file_content(f"replay_buffer_{position}")
        except FileNotFoundError:
            return None
        lengths = (self.arr_size, 4, 4, self.arr_size, 1, 22, 22)
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
            int.from_bytes(pieces[5], byteorder="big", signed=False),
            int.from_bytes(pieces[6], byteorder="big", signed=False),
        )

    async def get_file_content(self, file_name: str) -> bytes:
        async with aiofiles.open(f"{self.path}/{file_name}", mode="rb") as f:
            return await f.read()

    async def set_file_content(self, file_name: str, content: bytes) -> None:
        async with aiofiles.open(f"{self.path}/{file_name}", mode="wb") as f:
            return await f.write(content)


class DiskBackupProcess(Process):
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
        super(Process, self).__init__(daemon=True)
        self.disk = DiskBackup(obs_shape, version)

        self.in_queue: Queue = in_queue
        self.out_queue: Queue = out_queue

        self.size = size
        self.filled = False
        self.active = False
        self._write_pointer = 0
        self._read_pointer = 0
        self.batch_size = batch_size

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
                    coroutines.append(asyncio.create_task(self.store(self.write_pointer, experience)))
                if coroutines:
                    await asyncio.wait(coroutines)

            if self.out_queue.empty() and (self.filled or self._write_pointer > self._read_pointer + self.batch_size):
                self.out_queue.put_many([el for el in await self.prepare_batch() if el is not None])
                await asyncio.sleep(0.01)
            elif self._write_pointer == 0:
                await asyncio.sleep(0.01)

    async def prepare_batch(self):
        return await asyncio.gather(*(self.get(self.read_pointer) for _ in range(self.batch_size)))

    async def get(self, position: int) -> Experience | None:
        return await self.disk.get(position)

    async def store(self, position: int, experience: Experience):
        await self.disk.store(position, experience)
