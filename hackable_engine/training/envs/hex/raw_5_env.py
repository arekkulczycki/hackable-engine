# -*- coding: utf-8 -*-
from itertools import cycle
from random import randint, choice
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.envs.registration import register
from numpy import float32

from hackable_engine.training.envs.hex.base_raw_env import BaseRawEnv, MINUS_ONE, ONE, ZERO


class Raw5Env(BaseRawEnv):
    """"""

    ENV_NAME: str = "raw5hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 5, 5), dtype=float32
    )  # should be int8

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 5
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        # fmt: off
        self.OPENINGS = cycle(
            [
                "a1", "a2", "a3", "a4", "a5"
                "e1", "e2", "e3", "e4", "e5"
                "c2", "c4",
            ]
        )
        # fmt: on

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """"""

        super(BaseRawEnv, self).reset(seed=seed)

        self.render()

        # self.intermediate_rewards.clear()
        self.opp_model = choice(self.models)
        self.winner = None
        self.generations = 0

        # winner = self.controller.board.winner_no_turn()
        # if winner is not None:
        notation = next(self.OPENINGS)
        self.opening = notation
        self.controller.reset_board(
            notation=notation, size=self.BOARD_SIZE, init_move_stack=True
        )
        # n = randint(8, 16)
        # while True:
        #     self._prefill_board_randomly(n)
        #     if self.controller.board.turn != self.color:
        #         self._make_opponent_move()
        #
        #     try:
        #         self.initial_reward = self._get_intermediate_reward(len(self.controller.board.move_stack))
        #     except:
        #         self.controller.reset_board(
        #             notation=notation, size=self.BOARD_SIZE, init_move_stack=True
        #         )
        #         continue
        #     break
        if self.controller.board.turn != self.color:
            self._make_opponent_move()

        # self.initial_reward = self._get_intermediate_reward(len(self.controller.board.move_stack))

        self._prepare_child_moves()

        # must be last, because the policy should evaluate the first move candidate
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _prefill_board_randomly(self, n):
        for _ in range(n):
            self._make_random_move(self.controller.board)

    def _get_intermediate_reward(self, n_moves):
        return ZERO

    # def _on_stop_iteration(self) -> tuple[bool | None, float32]:
    #     winner, reward = super()._on_stop_iteration()
    #     return reward > 0, ONE if reward > self.initial_reward else MINUS_ONE

    def _make_opponent_move(self):
        # self._make_random_move(self.controller.board)
        self._make_logical_move(self.controller.board)
        # self._make_self_trained_move(
        #     self.controller.board, self.opp_model, not self.color
        # )


register(
    id="Raw5Env",
    entry_point="hackable_engine.training.envs.hex.raw_5_env:Raw5Env",
)
