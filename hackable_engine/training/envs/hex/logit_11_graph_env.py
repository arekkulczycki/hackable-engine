# -*- coding: utf-8 -*-
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

class Logit11GraphEnv(Logit7GraphEnv):
    """"""

    ENV_NAME = "logit9ghex"

    observation_space = gym.spaces.Box(
        0, 1, shape=(121, 9), dtype=FLOAT_TYPE
        # 0, 1, shape=(3, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8
    # action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)
    action_space = gym.spaces.Discrete(121)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 11
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11",
            "k1", "k2", "k3", "k4", "k5", "k6", "k7", "k8", "k9", "k10", "k11"
            "d2", "d10", "e2", "e10", "f2", "f10"
        ]
        # fmt: on

    def observation_from_board(self) -> np.ndarray:
        # for GraphGAT / GraphGINE
        return self.controller.board.get_hetero_graph_node_features_one_hot()

        # for GraphSG / GraphGIN
        # return self.controller.board.get_homo_graph_node_features_one_hot()

        # for CNN
        # return self.controller.board.as_matrix()


register(
    id="Logit11GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_11_graph_env:Logit11GraphEnv",
)
