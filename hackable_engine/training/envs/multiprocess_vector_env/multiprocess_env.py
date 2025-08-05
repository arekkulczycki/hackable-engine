from __future__ import annotations

from collections import deque
from math import sqrt
from multiprocessing import Lock, Process
from queue import Empty, Full
from time import sleep
from typing import Callable, Any

import numpy as np
from faster_fifo import Queue
from gymnasium import Env
from gymnasium.core import ActType
from numpy import ndarray

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.common.memory.adapters.shared_memory_adapter import (
    SharedMemoryAdapter,
)
from hackable_engine.training.envs.multiprocess_vector_env.batch_inference_vector_env import BatchInferenceVectorEnv
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
        self.local_env = make_env(-1, env_per_worker, color)
        # self.local_env.unwrapped._rewards.astype(FLOAT_TYPE, copy=False)
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

        self.buf_obs = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=FLOAT_TYPE)
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        # self.buf_blank = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=FLOAT_TYPE)
        # self.buf_infos = [{} for _ in range(self.num_envs)]
        self.buf_occupied_black_masks = np.zeros((self.num_envs, 22), dtype="uint8")  # TODO: S22 is just for 13x13
        self.buf_occupied_white_masks = np.zeros((self.num_envs, 22), dtype="uint8")  # TODO: S22 is just for 13x13

        self.time_queue = deque(maxlen=self.num_envs)
        self.return_queue = deque(maxlen=self.num_envs)
        """Cumulative episode return including illegal move penalties."""
        self.reward_queue = deque(maxlen=self.num_envs)
        """Actual rewards in completed episodes, i.e. excluding illegal move penalties."""
        self.winner_queue = deque(maxlen=self.num_envs)  # of np.float16 type
        """Tracking number of wins vs losses where illegal move is considered a loss."""
        self.legal_queue = deque(maxlen=self.num_envs)  # of np.float16 type
        """If the episode concluded with a legal move."""
        self.length_queue = deque(maxlen=self.num_envs)
        """Tracks lengths of episodes, i.e. number of moves of the agent."""
        self.action_queue = deque(maxlen=self.num_envs * 4)

    def get_progress_data(self) -> EnvProgressData:
        total_count = 0
        legal_count = 0
        length_sum = 0
        win_count = 0
        win_lengths_sum = 0
        win_squares_sum = 0
        loss_squares_sum = 0
        for is_win, length, legal in zip(self.winner_queue, self.length_queue, self.legal_queue):
            total_count += 1
            if legal:
                legal_count += 1
            length_sum += length
            if is_win:
                win_count += 1
                win_lengths_sum += length
                win_squares_sum += length**2
            else:
                loss_squares_sum += length**2

        loss_count = total_count - win_count
        win_length_mean = win_lengths_sum / win_count if win_count else 0
        loss_length_mean = (length_sum - win_lengths_sum) / loss_count
        win_variance = ((win_squares_sum - win_count * win_length_mean ** 2) / (win_count - 1)) if win_count > 1 else 0
        loss_variance = ((loss_squares_sum - loss_count * loss_length_mean ** 2) / (loss_count - 1)) if loss_count > 1 else 0

        return EnvProgressData(
            time_mean=np.sum(self.time_queue) / total_count,
            length_mean=length_sum / total_count,
            win_length_mean=win_length_mean,
            win_length_std=sqrt(win_variance) if win_variance else 0,
            loss_length_mean=loss_length_mean,
            loss_length_std=sqrt(loss_variance) if loss_variance else 0,
            return_mean=np.sum(self.return_queue) / total_count,
            reward_mean=np.sum(self.reward_queue) / legal_count,
            winner_mean=win_count / total_count,
            legal_mean=np.sum(self.legal_queue) / total_count,
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
    ) -> tuple[list[ndarray], tuple[int, ...], tuple[int, ...]]:  # type: ignore
        for process_id in range(self.num_workers):
            self.queues[process_id].put({"command": "reset"})

        pending = [True for _ in range(self.num_workers)]
        buf_occupied_black_masks = np.zeros((self.num_envs,), dtype=object)
        buf_occupied_white_masks = np.zeros((self.num_envs,), dtype=object)
        while any(pending):
            try:
                responses = self.parent_queue.get_many(block=True, timeout=1.0)
                # responses = (self.parent_queue.get(block=True, timeout=1.0),)
            except Empty:
                continue
            for response in responses:
                obs, info = response.get("reset", (None, None))
                if obs is None:
                    continue

                process_id = response["process_id"]
                # print(f"process {process_id} ready")
                start = process_id * self.env_per_worker
                stop = start + self.env_per_worker
                self.buf_obs[start:stop] = obs

                ocb_masks = np.ndarray(shape=(self.env_per_worker,), dtype=object, buffer=info["ocb"])
                ocw_masks = np.ndarray(shape=(self.env_per_worker,), dtype=object, buffer=info["ocw"])
                buf_occupied_black_masks[start:stop] = ocb_masks
                buf_occupied_white_masks[start:stop] = ocw_masks

                pending[process_id] = False

        # return np.split(self.buf_obs, self.num_envs, axis=0)
        return (
            [element.copy() for element in np.split(self.buf_obs, self.num_envs, axis=0)],
            tuple(int.from_bytes(b) for b in buf_occupied_black_masks),  # sending occupied masks in place of info
            tuple(int.from_bytes(b) for b in buf_occupied_white_masks),  # sending occupied masks in place of info
        )

    def step(
        self, actions: ActType
    ) -> tuple[list[ndarray], list[FLOAT_TYPE], list[bool], list[bool], tuple[int, ...], tuple[int, ...]]:
        self.send_actions(actions)
        return self.step_wait()

    def step_wait(
        self,
    ) -> tuple[
        list[ndarray],
        list[FLOAT_TYPE],
        list[bool],
        list[bool],
        tuple[int, ...],
        tuple[int, ...],
    ]:
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
            [element.copy() for element in np.split(self.buf_obs, self.num_envs, axis=0)],
            self.buf_rews.tolist(),  # .copy(),
            self.buf_dones.tolist(),  # .copy(),
            [False for _ in range(self.num_envs)],  # self.buf_blank.copy(),
            # [None for _ in range(self.num_envs)],  # deepcopy(self.buf_infos),
            tuple(int.from_bytes(b) for b in self.buf_occupied_black_masks),  # sending occupied masks in place of info
            tuple(int.from_bytes(b) for b in self.buf_occupied_white_masks),  # sending occupied masks in place of info
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
            ocb_masks = self._get_data(
                self.shm_data_key.format(i=process_id, t="ocb"),
                (self.env_per_worker, 22),  # TODO: S22 is just for 13x13
                dtype="uint8",
            )
            ocw_masks = self._get_data(
                self.shm_data_key.format(i=process_id, t="ocw"),
                (self.env_per_worker, 22),  # TODO: S22 is just for 13x13
                dtype="uint8",
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
            legal_info = self._get_data(
                self.shm_data_key.format(i=process_id, t="legal"),
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
            self.legal_queue.extend(legal_info[i])
            self.reward_queue.extend(reww_info[i])

        start = process_id * self.env_per_worker
        stop = start + self.env_per_worker
        self.buf_obs[start:stop] = obs
        self.buf_rews[start:stop] = rews
        self.buf_dones[start:stop] = dones
        self.buf_occupied_black_masks[start:stop] = ocb_masks
        self.buf_occupied_white_masks[start:stop] = ocw_masks
        # self.buf_infos[start:stop] = info

    def send_actions(self, action_list: ndarray) -> None:
        for process_id in range(self.num_workers):
            # Thread(target=self._send_action_to_process, args=(process_id, action_list)).start()
            self._send_action_to_process(process_id, action_list)

    def _send_action_to_process(self, process_id: int, action_list: ndarray):
        actions = action_list[process_id * self.env_per_worker : (process_id + 1) * self.env_per_worker]

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
        # color_ext = "black" if color else "white"
        # for model_version in ["gmm1", "gmm2", "sg"]:
        #     path = f"11_{color_ext}_{model_version}.onnx"
        #     models.append(
        #         InferenceSession(path, providers=["CPUExecutionProvider"])
        #         # InferenceSession(path, providers=["OpenVINOExecutionProvider"])
        #     )
        self.process_id = process_id
        self.env_per_worker = env_per_worker
        self.action_shape = action_shape
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.in_lock = in_lock
        self.out_lock = out_lock
        self.env: BatchInferenceVectorEnv = make_env(process_id, env_per_worker, color)
        # self.env.unwrapped._rewards = self.env.unwrapped._rewards.astype(FLOAT_TYPE)

        self.shm = SharedMemoryAdapter()
        self.shm_data_key = f"remote_env_{process_id}_{{t}}"
        self.shm_actions_key = "remote_env_{i}_actions".format(i=process_id)
        self.shm_obs_key = "remote_env_{i}_obs".format(i=process_id)
        self.shm_dones_key = "remote_env_{i}_dones".format(i=process_id)
        self.shm_rews_key = "remote_env_{i}_rews".format(i=process_id)
        self.shm_ocb_key = "remote_env_{i}_ocb".format(i=process_id)
        self.shm_ocw_key = "remote_env_{i}_ocw".format(i=process_id)

    def run(self):
        # from pyinstrument import Profiler  # pylint: disable=import-outside-toplevel
        # profiler = Profiler()
        # profiler.start()
        #
        # def before_exit(*_) -> None:
        #     profiler.stop()
        #     profiler.print(show_all=True)
        #
        #     sys.exit(0)
        # signal(SIGTERM, before_exit)

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
        # TODO: implement in the `BatchInferenceVectorEnv` an option to reset only subset of environments
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
            ocb_arr = np.zeros((self.env_per_worker, 22), dtype="uint8")
            ocw_arr = np.zeros((self.env_per_worker, 22), dtype="uint8")
            for i in range(self.env_per_worker):
                ocb_arr[i] = np.frombuffer(infos["ocb"][i], dtype="uint8")
                ocw_arr[i] = np.frombuffer(infos["ocw"][i], dtype="uint8")
            self._set_data(self.shm_ocb_key, ocb_arr)
            self._set_data(self.shm_ocw_key, ocw_arr)
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
            try:
                self._set_data(self.shm_data_key.format(t="legal"), infos["legal"].astype(np.float16))
            except KeyError:
                print(infos)
            self._set_data(self.shm_data_key.format(t="reww"), infos["reward"] * infos["legal"])
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
        self.in_queue: Queue = Queue(max_size_bytes=10 * 1024 * 1024)
        self.out_queue: Queue = Queue(max_size_bytes=10 * 1024 * 1024)

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
        return self.out_queue.get(block=True, timeout=15)

    def step(self, actions):
        self.in_queue.put({"command": "step", "actions": actions})
        return self.out_queue.get(block=True)

    def get_progress_data(self):
        self.in_queue.put({"command": "progress"})
        return self.out_queue.get(block=True)
