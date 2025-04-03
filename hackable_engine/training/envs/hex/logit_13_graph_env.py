# -*- coding: utf-8 -*-
from random import choices

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.logit_7_graph_env import Logit7GraphEnv

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)

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
            "f3", "g3", "h3", "i3", "f9", "g9", "h9", "i9"
        ]
        # fmt: on

    def observation_from_board(self) -> np.ndarray:
        # for GraphGAT / GraphGINE
        return self.controller.board.get_hetero_graph_node_features_one_hot()

        # for GraphSG / GraphGIN
        # return self.controller.board.get_homo_graph_node_features_one_hot()

        # for CNN
        # return self.controller.board.as_matrix()

    def _make_opponent_move(self, n_moves):
        win_percentage = (
            np.mean(self.results) if len(self.results) >= 4 else 0.9
        )
        if choices([True, False], weights=((1 - win_percentage) * 0.5 + 0.01, 0.49 + win_percentage * 0.5)):
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)


register(
    id="Logit13GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_13_graph_env:Logit13GraphEnv",
)
