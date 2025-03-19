# -*- coding: utf-8 -*-
from itertools import cycle

import gymnasium as gym
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)


class Raw9Env(BaseEnv):
    """"""

    ENV_NAME = "raw9env"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 9
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        self.AUXILIARY_REWARD_PER_MOVE = 0.5 * self.REWARDS[True] / (self.MAX_MOVES / 2)
        """Getting auxiliary reward on every move totals to a % of a win."""
        self.AUXILIARY_REWARD_PER_STEP = self.AUXILIARY_REWARD_PER_MOVE / self.MAX_MOVES
        # fmt: off
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
            "d2", "d8", "e2", "e8",
        ]
        # fmt: on


register(
    id="Raw9Env",
    entry_point="hackable_engine.training.envs.hex.raw_9_env:Raw9Env",
)
