# -*- coding: utf-8 -*-
from random import choices
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.logit_7_graph_env import Logit7GraphEnv

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)
ZERO_BYTES = (0).to_bytes(22)

class Logit13GraphEnv(Logit7GraphEnv):
    """"""

    ENV_NAME = "logit13ghex"

    observation_space = gym.spaces.Box(
        0, 1, shape=(169, 9), dtype=FLOAT_TYPE
        # 0, 1, shape=(3, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8
    # action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)
    action_space = gym.spaces.Discrete(169)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 13
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11", "a12", "a13",
            "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10", "m11", "m12", "m13",
            "c2", "c12", "d2", "d12", "e2", "e12", "f2", "f12", "g2", "g12", "h2", "h12", "i2", "i12", "k2", "k12",
            "b2", "l12", "f3", "g3", "h3", "i3", "f9", "g9", "h9", "i9"
        ]
        # fmt: on
        self.PENALTY_PER_ILLEGAL_MOVE = 0.5 * self.REWARDS[False] / (self.MAX_MOVES / 2)

        self.opponent_move: Move | None = None
        self.opponent_move_random: bool = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.array, dict[str, Any]]:
        """"""

        obs, info = super().reset(seed=seed)
        info["ocb"] = self.controller.board.occupied_co[False].to_bytes(22)
        info["ocw"] = self.controller.board.occupied_co[True].to_bytes(22)
        return obs, info

    def observation_from_board(self) -> np.ndarray:
        # for GNN
        return self.controller.board.get_hetero_graph_node_features_one_hot()

        # for CNN
        # return self.controller.board.as_matrix()

    def _make_opponent_move(self, n_moves):
        minimum_logical_moves = 0.25
        win_percentage = (
            np.mean(self.results) if len(self.results) >= 4 else minimum_logical_moves
        )
        random_move_weight = (1 - win_percentage) * (1 - minimum_logical_moves) + 0.01
        if choices((True, False), weights=(random_move_weight, 1 - random_move_weight))[0]:
            move = self._make_random_move(self.controller.board)
            self.opponent_move_random = True
        else:
            move = self._make_logical_move(self.controller.board)
            self.opponent_move_random = False
        self.opponent_move = move

    def step_from_preselected(self, move_position):
        move_pos_int = int(move_position)
        move = Move(mask=1 << move_pos_int, size=self.BOARD_SIZE)
        try:
            self.controller.board.push(move)
        except ValueError:
            # raise
            # print(f"attempting to push {move_position}", move.get_coord())
            self.winner = not self.color
            n_moves = self.MAX_MOVES - self.controller.board.unoccupied.bit_count()
            # self.reward = FLOAT_TYPE(max((MINUS_TWO + n_moves/(self.MAX_MOVES - 2 * self.BOARD_SIZE), MINUS_ONEHALF)))
            self.reward = FLOAT_TYPE(MINUS_TWO + n_moves/(self.MAX_MOVES - 2 * self.BOARD_SIZE))
            # TODO: verify if this idea works, penalty and returning back the same position (maybe stop after X attempts?)
            # self.reward = self.PENALTY_PER_ILLEGAL_MOVE
            return (
                self.obs,
                self.reward,
                True,
                False,
                {
                    "action": move_pos_int,
                    "winner": False,
                    "reward": self.reward,
                    "legal": False,
                    "ocb": ZERO_BYTES,
                    "ocw": ZERO_BYTES,
                    "om": -1,
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
                "legal": True,
                "ocb": self.controller.board.occupied_co[False].to_bytes(22) if winner is None else ZERO_BYTES,
                "ocw": self.controller.board.occupied_co[True].to_bytes(22) if winner is None else ZERO_BYTES,
                "om": self.opponent_move.c if not self.opponent_move_random else -1,
                # "opening": self.opening,
            },
        )


register(
    id="Logit13GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_13_graph_env:Logit13GraphEnv",
)
