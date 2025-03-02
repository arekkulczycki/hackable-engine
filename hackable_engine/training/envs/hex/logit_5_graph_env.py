# -*- coding: utf-8 -*-
from itertools import cycle
from random import randint, choice, choices
from typing import Any

import gymnasium as gym
from gymnasium.core import ObsType, RenderFrame
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv, MINUS_ONE, ONE, ZERO
import numpy as np


class Logit5GraphEnv(BaseEnv):
    """"""

    ENV_NAME: str = "logit5ghex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(25, 1), dtype=FLOAT_TYPE
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=FLOAT_TYPE)

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

    def step(self, action):
        # return self.step_from_logits(action)
        return self.step_from_preselected(action)

    def step_from_preselected(self, move_position):
        move_position = int(move_position)
        move = Move(mask=1 << move_position, size=self.BOARD_SIZE)
        try:
            # if self.winner is not None:
            #     return self.obs, MINUS_ONE, True, True, {}
            self.controller.board.push(move)
        except ValueError:
            # print(f"attempting to push {move_position}", move.get_coord())
            self.winner = True
            self.reward = MINUS_ONE
            return (
                self.obs,
                MINUS_ONE,
                True,
                True,
                {
                    "action": 0,
                    "winner": False,
                    "reward": MINUS_ONE,
                    # "opening": self.opening,
                },
            )

        winner, reward = self._get_winner_and_reward(
            len(self.controller.board.move_stack), with_iterations=False
        )

        self.winner = winner
        self.reward = reward
        self.obs = self.observation_from_board()
        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": 0,
                "winner": winner == self.color,
                "reward": reward if winner is not None else ZERO,
                # "opening": self.opening,
            },
        )

    def _get_intermediate_reward(self, n_moves):
        return ZERO

    def _quick_win_value(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        return ZERO
        # return ((max(0, (n_moves - 2 * self.BOARD_SIZE)) / self.MAX_MOVES) ** 2) * ONE

    def observation_from_board(self) -> np.ndarray:
        return self.controller.board.get_homo_graph_node_features()

    def _make_opponent_move(self, n_moves):
        # self._make_random_move(self.controller.board)
        # # self._make_logical_move(self.controller.board)
        win_percentage = (
            np.mean(self.results) if len(self.results) >= 4 else 0.9  # 0.4
        )
        # square = (1 - win_percentage) ** 2
        # if choices([True, False], weights=((1 - win_percentage) / 4, 0.75 + win_percentage/4)):
        if choices([True, False], weights=((1 - win_percentage), win_percentage)):
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)

    def render(self, mode="human", close=False) -> RenderFrame:
        # return super().render()
        return ""


register(
    id="Logit5GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_5_graph_env:Logit5GraphEnv",
)
