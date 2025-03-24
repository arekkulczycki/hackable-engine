# -*- coding: utf-8 -*-
from __future__ import annotations

from collections import deque
from multiprocessing import Lock, Process
from queue import Empty, Full
from time import sleep
from typing import Callable, Any

import numpy as np
from faster_fifo import Queue
from gymnasium import Env
from gymnasium.core import ActType
from gymnasium.vector.vector_env import VectorEnv
from numpy import ndarray

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.common.memory.adapters.shared_memory_adapter import (
    SharedMemoryAdapter,
)
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData


class MultiprocessEnv:
    def __init__(
        self,
        make_env: Callable[[int, int, bool], Env],
        num_workers: int,
        env_per_worker: int,
        color: bool = True,
        action_shape: tuple[int, ...] | None = None,
    ):
        self.make_local_env = make_env
        self.local_env = make_env(num_workers, env_per_worker, color)
        self.local_env.unwrapped._rewards.astype(FLOAT_TYPE, copy=False)
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.num_envs = num_workers * env_per_worker
        self.action_shape = action_shape or self.single_action_space.shape
        self.color = color

        self.shm = SharedMemoryAdapter()
        self.shm_data_key = "remote_env_{i}_{t}"

        self.queues = {i: Queue(max_size_bytes=10 * 1024 * 1024) for i in range(num_workers)}
        # self.queues: dict[int, Queue[dict[str, str]]] = {i: Queue() for i in range(num_workers)}
        self.read_locks = {i: Lock() for i in range(num_workers)}
        self.write_locks = {i: Lock() for i in range(num_workers)}
        self.parent_queue = Queue(max_size_bytes=num_workers * 10 * 1024 * 1024)
        # self.parent_queue: Queue[dict[str, Any]] = Queue()
        self.processes = {
            i: ProcessEnv(
                i,
                self.queues[i],
                self.parent_queue,
                self.write_locks[i],
                self.read_locks[i],
                make_env,
                env_per_worker,
                self.action_shape,
                color,
            )
            for i in range(num_workers)
        }
        for key, p in self.processes.items():
            p.start()

        self.buf_obs = np.zeros(
            (self.num_envs, *self.single_observation_space.shape), dtype=FLOAT_TYPE
        )
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_blank = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=FLOAT_TYPE)
        self.buf_infos = [{} for _ in range(self.num_envs)]

        self.time_queue = deque(maxlen=self.num_envs)
        self.return_queue = deque(maxlen=self.num_envs)
        self.reward_queue = deque(maxlen=self.num_envs)
        self.winner_queue = deque(maxlen=self.num_envs)  # of np.float16 type
        self.length_queue = deque(maxlen=self.num_envs)
        self.action_queue = deque(maxlen=self.num_envs * 4)

    def get_progress_data(self) -> EnvProgressData:
        return EnvProgressData(
            time_mean=np.mean(self.time_queue),
            length_mean=np.mean(self.length_queue),
            return_mean=np.mean(self.return_queue),
            reward_mean=np.mean(self.reward_queue),
            winner_mean=np.mean(self.winner_queue),
        )

    @property
    def single_observation_space(self):
        return self.local_env.single_observation_space

    @property
    def single_action_space(self):
        return self.local_env.single_action_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | None = None,  # TODO: implement an option to reset a subset
    ) -> list[ndarray]:  # type: ignore
        for process_id in range(self.num_workers):
            self.queues[process_id].put({"command": "reset"})

        pending = [True for _ in range(self.num_workers)]
        while any(pending):
            try:
                responses = self.parent_queue.get_many(block=True, timeout=1.0)
                # responses = (self.parent_queue.get(block=True, timeout=1.0),)
            except Empty:
                continue
            for response in responses:
                obs, _ = response.get("reset", (None, None))
                if obs is None:
                    continue

                process_id = response["process_id"]
                # print(f"process {process_id} ready")
                self.buf_obs[
                    process_id
                    * self.env_per_worker : (process_id + 1)
                    * self.env_per_worker
                ] = obs

                pending[process_id] = False

        # return np.split(self.buf_obs, self.num_envs, axis=0)
        return [element.copy() for element in np.split(self.buf_obs, self.num_envs, axis=0)]

    def step(
        self, actions: ActType
    ) -> tuple[list[ndarray], list[np.float32], list[bool], list[bool], list[None]]:
        self.send_actions(actions)
        return self.step_wait()

    def step_wait(self) -> tuple[list[ndarray], list[np.float32], list[bool], list[bool], list[None]]:
        # Wait for at least 1 env to be ready here
        # self.tr.diff()
        responses = []
        while len(responses) < self.num_workers:
            try:
                responses.extend(self.parent_queue.get_many(block=True, timeout=1.0))
                # responses.extend((self.parent_queue.get(block=True, timeout=1.0),))
            except Empty:
                continue

        # threads = []
        for response in responses:
            process_id = response["process_id"]
            has_episode = response["has_episode"]
            self._get_data_from_process(process_id, has_episode)
            # t = Thread(target=self._get_data_from_process, args=(process_id, has_episode))
            # t.start()
            # threads.append(t)
        # for t in threads:
        #     t.join()
        return (
            # np.split(self.buf_obs, self.num_envs, axis=0),#.copy(),
            [element.copy() for element in np.split(self.buf_obs, self.num_envs, axis=0)],
            self.buf_rews.tolist(),#.copy(),
            self.buf_dones.tolist(),#.copy(),
            [False for _ in range(self.num_envs)],  # self.buf_blank.copy(),
            [None for _ in range(self.num_envs)],  # deepcopy(self.buf_infos),
        )

    def _get_data_from_process(self, process_id: int, has_episode: bool):
        with self.read_locks[process_id]:
            obs = self._get_data(
                self.shm_data_key.format(i=process_id, t="obs"),
                (self.env_per_worker, *self.single_observation_space.shape),
                dtype=FLOAT_TYPE,
            )
            dones = self._get_data(
                self.shm_data_key.format(i=process_id, t="dones"),
                (self.env_per_worker,),
                dtype=bool,
            )
            rews = self._get_data(
                self.shm_data_key.format(i=process_id, t="rews"),
                (self.env_per_worker,),
                dtype=FLOAT_TYPE,
            )

            time_info = (
                self._get_data(
                    self.shm_data_key.format(i=process_id, t="time"),
                    (self.env_per_worker,),
                    dtype=FLOAT_TYPE,
                )
                if has_episode
                else None
            )
            len_info = (
                self._get_data(
                    self.shm_data_key.format(i=process_id, t="len"),
                    (self.env_per_worker,),
                    dtype=int,
                )
                if has_episode
                else None
            )
            rew_info = (
                self._get_data(
                    self.shm_data_key.format(i=process_id, t="rew"),
                    (self.env_per_worker,),
                    dtype=FLOAT_TYPE,
                )
                if has_episode
                else None
            )
            win_info = self._get_data(
                self.shm_data_key.format(i=process_id, t="win"),
                (self.env_per_worker,),
                dtype=np.float16,
            )
            reww_info = self._get_data(
                self.shm_data_key.format(i=process_id, t="reww"),
                (self.env_per_worker,),
                dtype=FLOAT_TYPE,
            )
        #     act_info = self._get_data(
        #         self.shm_data_key.format(i=process_id, t="act"),
        #         (self.env_per_worker,),
        #         dtype=np.int64,
        #     )
        #
        # self.action_queue.extend(act_info)

        for i in np.where(dones):
            if has_episode:
                self.time_queue.extend(time_info[i])
                self.length_queue.extend(len_info[i])
                self.return_queue.extend(rew_info[i])
            self.winner_queue.extend(win_info[i])
            self.reward_queue.extend(reww_info[i])

        start = process_id * self.env_per_worker
        stop = start + self.env_per_worker
        self.buf_obs[start:stop] = obs
        self.buf_rews[start:stop] = rews
        self.buf_dones[start:stop] = dones
        # self.buf_infos[start:stop] = info

    def send_actions(self, action_list: ndarray) -> None:
        for process_id in range(self.num_workers):
            # Thread(target=self._send_action_to_process, args=(process_id, action_list)).start()
            self._send_action_to_process(process_id, action_list)

    def _send_action_to_process(self, process_id: int, action_list: ndarray):
        actions = action_list[
            process_id * self.env_per_worker : (process_id + 1) * self.env_per_worker
        ]

        with self.write_locks[process_id]:
            try:
                self._set_data(
                    self.shm_data_key.format(i=process_id, t="actions"),
                    np.array(actions, dtype=FLOAT_TYPE),
                )
            except:
                print(action_list)
                print(process_id, self.env_per_worker, len(action_list))
                raise
        self.queues[process_id].put({"command": "step"})

    def try_reset(self, process_id: int) -> None:
        self.queues[process_id].put({"command": "reset"})

    def close(self):
        self.stop()

    def stop(self) -> None:
        for process_id, p in self.processes.items():
            p.terminate()

    def _get_data(self, key: str, shape: tuple[int, ...], dtype) -> ndarray:
        return np.ndarray(shape=shape, dtype=dtype, buffer=self.shm.get(key))

    def _set_data(self, key: str, data: ndarray) -> None:
        self.shm.set(key, data.tobytes())


class ProcessEnv(Process):

    def __init__(
        self,
        process_id: int,
        in_queue: Queue[dict[str, str]],
        out_queue: Queue[dict[str, Any]],
        in_lock: Lock,
        out_lock: Lock,
        make_env,
        env_per_worker,
        action_shape,
        color,
    ):
        super().__init__(daemon=True)
        # models = []
        # color_ext = "Black" if color else "White"
        # for model_version in ["A", "B", "C", "D", "E", "F"]:
        #     path = f"Hex9{color_ext}{model_version}.onnx"
        #     models.append(
        #         ort.InferenceSession(path, providers=["OpenVINOExecutionProvider"])
        #     )
        self.process_id = process_id
        self.env_per_worker = env_per_worker
        self.action_shape = action_shape
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_lock = in_lock
        self.out_lock = out_lock
        self.env: VectorEnv = make_env(process_id, env_per_worker, color, [None])
        self.env.unwrapped._rewards = self.env.unwrapped._rewards.astype(FLOAT_TYPE)

        self.shm = SharedMemoryAdapter()
        self.shm_data_key = f"remote_env_{process_id}_{{t}}"
        self.shm_actions_key = "remote_env_{i}_actions".format(i=process_id)
        self.shm_obs_key = "remote_env_{i}_obs".format(i=process_id)
        self.shm_dones_key = "remote_env_{i}_dones".format(i=process_id)
        self.shm_rews_key = "remote_env_{i}_rews".format(i=process_id)

    def run(self):
        should_get = True
        while True:
            if should_get:
                try:
                    d = self.in_queue.get(block=True, timeout=1.0)
                except Empty:
                    continue
            if d["command"] == "reset":
                try:
                    self.reset()
                except Full:
                    sleep(1)
                    should_get = False
                    print(f"{self.process_id}: out queue full")
                    continue
                else:
                    should_get = True
            elif d["command"] == "step":
                try:
                    self.step()
                except Full:
                    sleep(1)
                    should_get = False
                    print(f"{self.process_id}: out queue full")
                    continue
                else:
                    should_get = True

    def reset(self):
        # TODO: implement in the VectorEnv an option to reset only subset of environments
        self.out_queue.put({"process_id": self.process_id, "reset": self.env.reset()})

    def step(self):
        with self.in_lock:
            actions = self._get_actions()

        obs, rews, dones, _, infos = self.env.step(actions)
        has_episode = "episode" in infos
        with self.out_lock:
            self._set_data(self.shm_obs_key, obs)
            self._set_data(self.shm_rews_key, rews)
            self._set_data(self.shm_dones_key, dones)
            if has_episode:
                self._set_data(
                    self.shm_data_key.format(t="time"),
                    infos["episode"]["t"].astype(FLOAT_TYPE),
                )
                self._set_data(self.shm_data_key.format(t="len"), infos["episode"]["l"])
                self._set_data(
                    self.shm_data_key.format(t="rew"),
                    infos["episode"]["r"].astype(FLOAT_TYPE),
                )
            self._set_data(self.shm_data_key.format(t="win"), infos["winner"].astype(np.float16))
            self._set_data(self.shm_data_key.format(t="reww"), infos["reward"])
            self._set_data(self.shm_data_key.format(t="act"), infos["action"])

        self.out_queue.put({"process_id": self.process_id, "has_episode": has_episode})

    def _get_actions(self) -> ndarray:
        return np.ndarray(
            shape=(self.env_per_worker, *self.action_shape),
            dtype=FLOAT_TYPE,
            buffer=self.shm.get(self.shm_actions_key),
        )

    def _set_data(self, key: str, data: ndarray) -> None:
        self.shm.set(key, data.tobytes())


class MultiprocessEnvRunner(Process):

    def __init__(self, *args, **kwargs):
        super().__init__(daemon=True)
        self.env = MultiprocessEnv(*args, **kwargs)
        # TODO: must expose the monitoring queues, maybe through shm?
        self.in_queue: Queue = Queue()
        self.out_queue: Queue = Queue()

    def run(self):
        should_get = True
        while True:
            if should_get:
                try:
                    d = self.in_queue.get(block=True, timeout=1.0)
                except Empty:
                    continue
            if d["command"] == "reset":
                try:
                    self.out_queue.put(self.env.reset())
                except Full:
                    sleep(1)
                    should_get = False
                    print(f"runner: out queue full")
                    continue
                else:
                    should_get = True
            elif d["command"] == "step":
                try:
                    self.out_queue.put(self.env.step(d["actions"]))
                except Full:
                    sleep(1)
                    should_get = False
                    print(f"runner: out queue full")
                    continue
                else:
                    should_get = True
            elif d["command"] == "progress":
                try:
                    self.out_queue.put(self.env.get_progress_data())
                except Full:
                    sleep(1)
                    should_get = False
                    print(f"runner: out queue full")
                    continue
                else:
                    should_get = True

    def reset(self, *args, **kwargs):
        self.in_queue.put({"command": "reset"})
        return self.out_queue.get(block=True)

    def step(self, actions):
        self.in_queue.put({"command": "step", "actions": actions})
        return self.out_queue.get(block=True)

    def get_progress_data(self):
        self.in_queue.put({"command": "progress"})
        return self.out_queue.get(block=True)
