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
        # fmt: off
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
            "d2", "d8", "e2", "e8",
        ]
        # fmt: on

    # def _prefill_board_randomly(self):
    #     for _ in range(40):
    #         self._make_random_move(self.controller.board)

    def _get_intermediate_reward(self, n_moves):
        return ZERO

    def _principle_bonus(self, color) -> FLOAT_TYPE:
        """Balanced distribution of stones is always preferred. 3/4 of board size is the highest value."""

        imbalance, central_imbalance = self.controller.board.get_imbalance(color)
        return (
            FLOAT_TYPE(self.BOARD_SIZE * 3 / 4 - imbalance - central_imbalance) * FLOAT_TYPE(0.015 / (3 / 4 * 9))
        )

    def _make_opponent_move(self, n_moves):
        if n_moves <= 48:
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)
        # self._make_self_trained_move(
        #     self.controller.board, self.opp_model, not self.color
        # )


register(
    id="Raw9Env",
    entry_point="hackable_engine.training.envs.hex.raw_9_env:Raw9Env",
)
