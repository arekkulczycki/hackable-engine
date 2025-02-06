# -*- coding: utf-8 -*-
from random import random
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import register
from numpy import float32
from stable_baselines3.common.policies import ActorCriticPolicy

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)


class DummyEnv(gym.Env):
    """"""

    ENV_NAME: str = "dummy"
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    """
    Maximum and minimum sum aggregated over an entire episode, not just the final reward. 
    """  # TODO: aggregated really?

    observation_space = gym.spaces.Box(
        -1, 1, shape=(2, 2), dtype=float32
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=float32)

    winner: Optional[bool]
    obs: th.Tensor

    policy: Optional[ActorCriticPolicy] = None

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.target_position: list[float32] = [float32(0.0), float32(0.0)]
        self.agent_position: list[float32] = [float32(0.0), float32(0.0)]

        self.use_intermediate_reward_absolute = False
        self.use_intermediate_reward_relative = False
        self.max_distance = float32(2.5)
        self.step_length = float32(0.0666)

        self.last_distance = 0.0
        self.steps_taken = 0

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        msg = (
            f"starting positions: {self.agent_position} -> {self.target_position}"
            f"\nsteps taken: {self.steps_taken}, finished at: {self.last_distance}"
        )
        print(msg)
        return msg

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """"""

        super().reset(seed=seed)

        self.render()

        self.steps_taken = 0

        while True:
            self.target_position = [float32(random() * 2 - 1), float32(random() * 2 - 1)]
            self.agent_position = [float32(random() * 2 - 1), float32(random() * 2 - 1)]
            self.last_distance = self._get_distance()
            if self.last_distance > self.step_length:
                break

        self.obs = self.observation()
        return self.obs, {}

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """"""

        direction_degrees = action[0] * 180 + 180
        direction_radians = np.deg2rad(direction_degrees)
        self.agent_position[0] += self.step_length * np.cos(direction_radians)
        self.agent_position[1] += self.step_length * np.sin(direction_radians)

        self.obs = self.observation()
        distance = self._get_distance()
        reward = self._get_reward(distance)
        self.last_distance = distance
        self.steps_taken += 1

        return (
            self.obs,
            reward,
            reward in (ONE, MINUS_ONE),
            False,
            {
                "action": direction_degrees,
                "winner": reward if reward in [ONE, MINUS_ONE] else ZERO,
                "reward": reward,
                "opening": None,
            },
        )

    def _get_reward(self, distance) -> float32:
        if distance > self.max_distance or self.steps_taken > 500:
            # exceeded distance limit
            return MINUS_ONE
        if distance < self.step_length:
            # reached destination
            return ONE

        if self.use_intermediate_reward_absolute:
            return np.tanh((self.max_distance / distance) - 2) / 100
        elif self.use_intermediate_reward_relative:
            # return (self.last_distance - distance) / 10  # negative if increased distance
            if self.last_distance > distance:  # only positive feedback
                return (self.last_distance - distance) / 10
        return ZERO

    def _get_distance(self):
        return np.sqrt(
            (self.target_position[0] - self.agent_position[0]) ** 2
            + (self.target_position[1] - self.agent_position[1]) ** 2
        )

    def _make_opponent_move(self):
        self._make_random_move(self.controller.board)

    def _make_random_move(self):
        """"""

    def observation(self) -> th.Tensor:
        """Degrees angle toward target"""

        # return th.Tensor([self.target_position, self.agent_position])
        return th.maximum(
            th.minimum(
                th.Tensor([self.target_position, self.agent_position]), th.Tensor([1])
            ),
            th.Tensor([-1]),
        )

    def summarize(self):
        """"""

        print(
            f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}"
        )


register(
    id="DummyEnv",
    entry_point="hackable_engine.training.envs.dummy_env:DummyEnv",
)
