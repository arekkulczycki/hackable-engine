# -*- coding: utf-8 -*-
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from nptyping import Int8, NDArray, Shape
from numpy import int8, eye

from arek_chess.training.envs.hex.raw_9_env import Raw9Env


class Raw9x9Env(Raw9Env):
    """"""

    BOARD_SIZE: int = 9
    observation_space = gym.spaces.MultiBinary(BOARD_SIZE ** 2 * 2)

    @staticmethod
    def observation_from_board(board) -> NDArray[Shape["162"], Int8]:
        local: NDArray[Shape["81"], Int8] = board.get_neighbourhood(
            9, should_suppress=True
        ).flatten()
        # fmt: off
        return eye(3, dtype=int8)[local][:, 1:].flatten()  # dummy encoding - 2 columns of 0/1 values, 1 column dropped
        # fmt: on