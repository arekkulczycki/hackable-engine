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
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)

class Logit9GraphEnv(BaseEnv):
    """"""

    ENV_NAME = "logit9ghex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(81, 1), dtype=FLOAT_TYPE
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 9
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
            "d2", "d8", "e2", "e8",
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
            n_moves = self.MAX_MOVES - self.controller.board.unoccupied.bit_count()
            self.reward = max((MINUS_ONE - (1 - n_moves/self.MAX_MOVES), MINUS_ONEHALF))
            return (
                self.obs,
                self.reward,
                True,
                True,
                {
                    "action": 0,
                    "winner": False,
                    "reward": self.reward,
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
        if choices([True, False], weights=((1 - win_percentage) * 99/100, 0.01 + win_percentage)):
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)

    def _prepare_child_moves(self) -> None:
        return None

    def observation_from_board(self) -> np.ndarray:
        return self.controller.board.get_homo_graph_node_features()

    def render(self, mode="human", close=False):
        return super().render()
        # return ""


register(
    id="Logit7GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_7_graph_env:Logit7GraphEnv",
)
