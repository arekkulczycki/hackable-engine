# -*- coding: utf-8 -*-
from random import choices

import gymnasium as gym
import numpy as np
from torch.nn import functional as F
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.seq_7_env import Seq7Env

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)

class Logit7GraphEnv(Seq7Env):
    """"""

    ENV_NAME = "logit7ghex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(49, 1), dtype=FLOAT_TYPE
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

    def reset(
        self,
        *,
        seed = None,
        options = None,
    ):
        obs, _ = super().reset(seed=seed, options=options)
        return obs, {
            "action": 0,
            "winner": None,
            "reward": FLOAT_TYPE(0.0),
        }

    def step(self, action):
        # return self.step_from_logits(action)
        return self.step_from_preselected(action)

    def step_from_preselected(self, move_position):
        move_pos_int = int(move_position)
        move = Move(mask=1 << move_pos_int, size=self.BOARD_SIZE)
        try:
            # if self.winner is not None:
            #     return self.obs, MINUS_ONE, True, True, {}
            self.controller.board.push(move)
        except ValueError:
            # print(f"attempting to push {move_position}", move.get_coord())
            self.winner = not self.color
            n_moves = self.MAX_MOVES - self.controller.board.unoccupied.bit_count()
            # self.reward = FLOAT_TYPE(max((MINUS_TWO + n_moves/(self.MAX_MOVES - 2 * self.BOARD_SIZE), MINUS_ONEHALF)))
            self.reward = FLOAT_TYPE(MINUS_TWO + n_moves/(self.MAX_MOVES - 2 * self.BOARD_SIZE))
            return (
                self.obs,
                self.reward,
                True,
                False,
                {
                    "action": move_pos_int,
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
        # return super().render()
        return ""

    def _get_intermediate_reward(self, n_moves):
        # return FLOAT_TYPE(self._get_intermediate_reward_relative(n_moves))
        # win_percentage = (
        #     np.mean(self.results) if len(self.results) >= 4 else 0.9  # 0.4
        # )
        # if win_percentage > 0.9:
        #     return FLOAT_TYPE(self._get_intermediate_reward_relative(n_moves))
        return ZERO

    # def _quick_win_value(self, n_moves: int) -> float:
    #     """The more moves are played the higher the punishment."""
    #
    #     return ZERO
    #     # return ((max(0, (n_moves - 2 * self.BOARD_SIZE)) / self.MAX_MOVES) ** 2) * ONE



register(
    id="Logit7GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_7_graph_env:Logit7GraphEnv",
)
