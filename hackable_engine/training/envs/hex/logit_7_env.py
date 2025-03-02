# -*- coding: utf-8 -*-
from random import choices

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv
from hackable_engine.training.envs.hex.seq_7_env import Seq7Env

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)


class Logit7Env(BaseEnv):
    """"""

    ENV_NAME = "logit7hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 7, 7), dtype=FLOAT_TYPE
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(49,), dtype=FLOAT_TYPE)

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

    def step(self, action):
        # return self.step_from_logits(action)
        return self.step_from_preselected(action)

    def step_from_preselected(self, move_position):
        move = Move(mask=1 << int(move_position), size=self.BOARD_SIZE)
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
                MINUS_TWO,
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
            self.MAX_MOVES - self.controller.board.unoccupied.bit_count(), with_iterations=False
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

    def _prepare_child_moves(self) -> None:
        return None

    def render(self, mode="human", close=False):
        return super().render()
        # return ""


register(
    id="Logit7Env",
    entry_point="hackable_engine.training.envs.hex.logit_7_env:Logit7Env",
)
