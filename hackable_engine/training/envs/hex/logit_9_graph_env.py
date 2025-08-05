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

class Logit9GraphEnv(Logit7GraphEnv):
    """"""

    ENV_NAME = "logit9ghex"

    observation_space = gym.spaces.Box(
        0, 1, shape=(81, 9), dtype=FLOAT_TYPE
        # 0, 1, shape=(3, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8
    # action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)
    action_space = gym.spaces.Discrete(81)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 9
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
            "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
            "c2", "c8", "d2", "d8", "e2", "e8", "f2", "f8", "g2", "g8",
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
    id="Logit9GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_9_graph_env:Logit9GraphEnv",
)
