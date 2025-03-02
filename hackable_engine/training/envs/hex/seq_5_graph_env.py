# -*- coding: utf-8 -*-
from itertools import cycle
from random import randint, choice, choices
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv, MINUS_ONE, ONE, ZERO
import numpy as np

from hackable_engine.training.hyperparams import N_ENVS


class Seq5GraphEnv(BaseEnv):
    """"""

    ENV_NAME: str = "seq5hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(25, 1), dtype=FLOAT_TYPE
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
        if self.did_force_stop:
            distance = self._get_distance_score(n_moves, early_finish=False)
            if (distance > 0 and self.color) or (distance < 0 and not self.color):
                return FLOAT_TYPE(
                    4 * self.AUXILIARY_REWARD_PER_MOVE
                )  # getting this reward on every move totals to 0.5
            else:
                # penalty is twice smaller, if losing we want to incentivise a long game
                return FLOAT_TYPE(-self.AUXILIARY_REWARD_PER_MOVE)
        return ZERO

    def observation_from_board(self) -> np.ndarray:
        return self.controller.board.get_homo_graph_node_features()

    def _make_opponent_move(self, n_moves):
        self._make_random_move(self.controller.board)
        # self._make_logical_move(self.controller.board)
        # win_percentage = (
        #     np.mean(self.results) if len(self.results) >= N_ENVS / 2 else 1  # 0.4
        # )
        # # square = (1 - win_percentage) ** 2
        # if choices([True, False], weights=(1 - win_percentage, win_percentage)):
        #     self._make_random_move(self.controller.board)
        # else:
        #     self._make_logical_move(self.controller.board)


register(
    id="Seq5GraphEnv",
    entry_point="hackable_engine.training.envs.hex.seq_5_graph_env:Seq5GraphEnv",
)
