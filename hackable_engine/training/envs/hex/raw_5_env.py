# -*- coding: utf-8 -*-
from itertools import cycle
from random import randint, choice
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_raw_env import BaseRawEnv, MINUS_ONE, ONE, ZERO


class Raw5Env(BaseRawEnv):
    """"""

    ENV_NAME: str = "raw5hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 5, 5), dtype=FLOAT_TYPE
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 5
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        # fmt: off
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5"
            "e1", "e2", "e3", "e4", "e5"
            "c2", "c4",
        ]
        # fmt: on

    def _get_intermediate_reward(self, n_moves):
        return ZERO

    # def _on_stop_iteration(self) -> tuple[bool | None, FLOAT_TYPE]:
    #     winner, reward = super()._on_stop_iteration()
    #     return reward > 0, ONE if reward > self.initial_reward else MINUS_ONE

    def _make_opponent_move(self, n_moves):
        # self._make_random_move(self.controller.board)
        self._make_logical_move(self.controller.board)
        # self._make_self_trained_move(
        #     self.controller.board, self.opp_model, not self.color
        # )


register(
    id="Raw5Env",
    entry_point="hackable_engine.training.envs.hex.raw_5_env:Raw5Env",
)
