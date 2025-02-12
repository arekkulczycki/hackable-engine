# -*- coding: utf-8 -*-
from collections import deque
from itertools import cycle

import gymnasium as gym
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_raw_env import BaseRawEnv
import numpy as np
import torch as th

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)


class Raw7Env(BaseRawEnv):
    """"""

    ENV_NAME = "raw9env"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(49,), dtype=FLOAT_TYPE
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 7
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        # fmt: off
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7",
            "g1", "g2", "g3", "g4", "g5", "g6", "g7",
            "d2", "d6",
        ]
        # fmt: on

    def _get_intermediate_reward(self, n_moves):
        return ZERO
        # if n_moves <= 24:
        #     score = self._get_distance_score_perf(n_moves, early_finish=False)
        # else:
        #     score = self._get_distance_score(n_moves, early_finish=False)
        # relative_score = score - self.last_intermediate_score
        # self.last_intermediate_score = score
        #
        # if not self.color:
        #     relative_score *= MINUS_ONE
        #
        # return relative_score

    @staticmethod
    def observation_from_board(board) -> np.ndarray:
        return board.get_graph_node_features()


register(
    id="Raw7Env",
    entry_point="hackable_engine.training.envs.hex.raw_7_env:Raw7Env",
)
