# -*- coding: utf-8 -*-
from random import choices

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv, ZERO
from hackable_engine.training.hyperparams import N_ENVS, BATCH_SIZE


class Seq5Env(BaseEnv):
    """"""

    ENV_NAME: str = "seq5hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 5, 5), dtype=FLOAT_TYPE
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 5
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.AUXILIARY_REWARD_PER_MOVE = 0.5 * self.REWARDS[True] / (self.MAX_MOVES / 2)
        """Getting auxiliary reward on every move totals to a % of a win."""
        self.AUXILIARY_REWARD_PER_STEP = self.AUXILIARY_REWARD_PER_MOVE / self.MAX_MOVES
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
                    self.AUXILIARY_REWARD_PER_MOVE
                )
            else:
                return FLOAT_TYPE(-self.AUXILIARY_REWARD_PER_MOVE)
        return ZERO

    def _get_intersequence_reward(self, action_score):
        return ZERO

    def _quick_win_value(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        return ZERO
        # return ((max(0, (n_moves - 2 * self.BOARD_SIZE)) / self.MAX_MOVES) ** 2) * ONE


    def _make_opponent_move(self, n_moves):
        # self._make_random_move(self.controller.board)
        # self._make_logical_move(self.controller.board)
        win_percentage = (
            np.mean(self.results) if len(self.results) >= 4 else 0.9  # 0.4
        )
        # square = (1 - win_percentage) ** 2
        if choices([True, False], weights=(0.05 + (1 - win_percentage) / 4, 0.7 + win_percentage/4)):
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)


register(
    id="Seq5Env",
    entry_point="hackable_engine.training.envs.hex.seq_5_env:Seq5Env",
)
