# -*- coding: utf-8 -*-
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.seq_7_env import Seq7Env

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)


class Seq7GraphEnv(Seq7Env):
    """"""

    ENV_NAME = "seq7ghex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(49, 1), dtype=FLOAT_TYPE
    )  # should be int8

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

    def observation_from_board(self) -> np.ndarray:
        return self.controller.board.get_homo_graph_node_features()


register(
    id="Raw7GraphEnv",
    entry_point="hackable_engine.training.envs.hex.raw_7_graph_env:Raw7GraphEnv",
)
