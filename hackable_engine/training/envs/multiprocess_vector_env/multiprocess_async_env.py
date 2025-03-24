# -*- coding: utf-8 -*-
from threading import Thread
from typing import Callable, Any

import janus
import numpy as np
from gymnasium import Env

from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_env import (
    MultiprocessEnv,
    EnvProgressData,
)


class MultiprocessAsyncEnv:
    def __init__(
        self,
        make_env: Callable[[int, int, bool], Env],
        num_workers: int,
        env_per_worker: int,
        color: bool = True,
        action_shape: tuple[int, ...] | None = None,
    ):
        # self.env = MultiprocessEnvRunner(
        self.env = MultiprocessEnv(
            make_env, num_workers, env_per_worker, color, action_shape
        )
        # self.env.start()
        self.in_queue = janus.Queue(maxsize=num_workers * env_per_worker)
        self.out_queue = janus.Queue(maxsize=num_workers * env_per_worker)
        worker = Thread(
            target=self.work,
            args=(self.in_queue.sync_q, self.out_queue.sync_q),
            daemon=True,
        )
        worker.start()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | None = None,  # TODO: implement an option to reset a subset
    ) -> list[np.ndarray]:  # type: ignore
        return self.env.reset(seed=seed, options=options, env_ids=env_ids)

    async def step(
        self, actions
    ) -> tuple[np.ndarray, list[np.float32], list[bool], list[bool], list[None]]:
        await self.out_queue.async_q.put(actions)
        return await self.in_queue.async_q.get()

    def get_progress_data(self) -> EnvProgressData:
        return self.env.get_progress_data()

    def work(self, in_queue, out_queue):
        while True:
            in_queue.put(self.env.step(out_queue.get()))

    @property
    def time_queue(self):
        return self.env.time_queue

    @property
    def return_queue(self):
        return self.env.return_queue

    @property
    def reward_queue(self):
        return self.env.reward_queue

    @property
    def winner_queue(self):
        return self.env.winner_queue

    @property
    def length_queue(self):
        return self.env.length_queue

    @property
    def action_queue(self):
        return self.env.action_queue

    @property
    def single_observation_space(self):
        return self.env.single_observation_space
