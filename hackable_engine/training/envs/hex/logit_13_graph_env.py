from random import choices, sample, choice
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.envs.util import RealTimeMeanVariance
from onnxruntime import InferenceSession

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)
ZERO_BYTES = (0).to_bytes(22)


class Logit13GraphEnv(Logit11GraphEnv):
    """"""

    ENV_NAME = "logit13ghex"

    observation_space = gym.spaces.Box(
        0,
        1,
        shape=(169, 9),
        dtype=FLOAT_TYPE,
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
        # fmt: off
        self.OPENINGS = [
            "a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","a12","a13",
            "m1","m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12","m13",
            "c2","c12","d2","d12","e2","e12","f2","f12","g2","g12","h2","h12","i2","i12","k2","k12",
            "b2","l12","f3","g3","h3","i3","f11","g11","h11","i11",
        ]  # 52 openings
        # fmt: on

        self.rtm = RealTimeMeanVariance()
        opp_color_text = "black" if self.color else "white"
        self.opp_ort_session = None if self.models is None else InferenceSession(f"{self.BOARD_SIZE}_{opp_color_text}_{choice(self.models)}.onnx", providers=["CPUExecutionProvider"])


register(
    id="Logit13GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_13_graph_env:Logit13GraphEnv",
)
