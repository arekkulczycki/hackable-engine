# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from collections import deque

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium.vector.vector_env import ArrayType, VectorEnv, VectorWrapper
from gymnasium.wrappers.vector import RecordEpisodeStatistics

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.hyperparams import N_ENVS, N_STEPS


class EpisodeStats(RecordEpisodeStatistics):

    def __init__(
        self,
        env: VectorEnv,
        stats_key: str = "episode",
        is_multiprocessed: bool = False,
    ):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.stats_key = stats_key
        self.is_multiprocessed = is_multiprocessed

        self.episode_count = 0

        self.episode_start_times: np.ndarray = np.zeros((), dtype=FLOAT_TYPE)
        self.episode_returns: np.ndarray = np.zeros((), dtype=FLOAT_TYPE)
        self.episode_lengths: np.ndarray = np.zeros((), dtype=int)
        self.prev_dones: np.ndarray = np.zeros((), dtype=bool)

        self.time_queue = deque(maxlen=N_ENVS * 3)
        self.return_queue = deque(maxlen=N_ENVS * 3)
        self.reward_queue = deque(maxlen=N_ENVS * 3)
        self.length_queue = deque(maxlen=N_ENVS * 3)
        self.winner_queue = deque(maxlen=N_ENVS * 3)
        self.action_queue = deque(maxlen=N_STEPS)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict | None = None,
    ):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        obs, info = super().reset(seed=seed, options=options)

        # self.episode_start_times = np.full(self.num_envs, time.perf_counter())
        self.episode_returns = np.zeros(self.num_envs, dtype=FLOAT_TYPE)
        self.episode_lengths = np.zeros(self.num_envs, dtype=int)
        self.prev_dones = np.zeros(self.num_envs, dtype=bool)

        return obs, info

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict]:
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        self.episode_returns[self.prev_dones] = 0
        self.episode_lengths[self.prev_dones] = 0
        self.episode_start_times[self.prev_dones] = time.perf_counter()  #int(time.perf_counter() * 100) / 100
        self.episode_returns[~self.prev_dones] += rewards[~self.prev_dones]
        self.episode_lengths[~self.prev_dones] += 1
        if not self.is_multiprocessed:
            self.action_queue.extend(infos["action"])

        self.prev_dones = dones = np.logical_or(terminations, truncations)
        num_dones = np.sum(dones)

        if num_dones:
            self.episode_count += num_dones
            episode_fps = self.episode_lengths / (time.perf_counter() - self.episode_start_times)

            if self.is_multiprocessed:
                infos[self.stats_key] = {
                    "r": np.where(dones, self.episode_returns, 0.0),
                    "l": np.where(dones, self.episode_lengths, 0),
                    "t": np.where(dones, episode_fps, 0.0),
                }
                # infos[f"_{self._stats_key}"] = dones
            else:
                for i in np.where(dones):
                    self.time_queue.extend(episode_fps[i])
                    self.return_queue.extend(self.episode_returns[i])
                    self.length_queue.extend(self.episode_lengths[i])
                    self.winner_queue.extend(infos["winner"][i])  # x[~np.isnan(x)]
                    self.reward_queue.extend(infos["reward"][i])  # x[~np.isnan(x)]

        return (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        )
