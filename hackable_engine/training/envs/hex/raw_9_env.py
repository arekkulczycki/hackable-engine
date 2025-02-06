# -*- coding: utf-8 -*-
from itertools import cycle

import gymnasium as gym
from gymnasium.envs.registration import register
from numpy import float32

from hackable_engine.training.envs.hex.base_raw_env import BaseRawEnv

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)


class Raw9Env(BaseRawEnv):
    """"""

    ENV_NAME = "raw9env"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 9, 9), dtype=float32
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 9
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        # fmt: off
        self.OPENINGS = cycle(
            [
                "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
                "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
                "d2", "d8", "e2", "e8",
            ]
        )
        # fmt: on

    def _prefill_board_randomly(self):
        for _ in range(40):
            self._make_random_move(self.controller.board)

    # def _get_intermediate_reward(self, n_moves):
    #     return ZERO

    def _principle_bonus(self, color) -> float32:
        """Balanced distribution of stones is always preferred. 3/4 of board size is the highest value."""

        imbalance, central_imbalance = self.controller.board.get_imbalance(color)
        return (
            float32(self.BOARD_SIZE * 3 / 4 - imbalance - central_imbalance) * float32(0.015 / (3 / 4 * 9))
        )

    def _make_opponent_move(self):
        # self._make_random_move(self.controller.board)
        self._make_logical_move(self.controller.board)
        # self._make_self_trained_move(
        #     self.controller.board, self.opp_model, not self.color
        # )


register(
    id="Raw9Env",
    entry_point="hackable_engine.training.envs.hex.raw_9_env:Raw9Env",
)
