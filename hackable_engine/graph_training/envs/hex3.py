# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import cycle
from typing import (
    DefaultDict,
    Dict,
    Optional,
    Tuple,
)

import gymnasium
from nptyping import NDArray
from numpy import asarray, float32

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

# fmt: off
BOARD_LINKS: NDArray = asarray([
    (0, 1), (1, 2), (3, 4), (4, 5), (6, 7), (7, 8),
    (0, 3), (3, 6), (1, 4), (4, 7), (2, 5), (5, 8),
    (1, 3), (2, 4), (4, 6), (5, 7)
])
# fmt: on

openings = cycle(["a3", "b3", "c1", "a2"])


class Hex3(Raw9Env):
    """"""

    BOARD_SIZE: int = 3
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    num_nodes = 9  # 25, 49, n*n
    num_edges = (
        16  # 4*5 + 9*4 = 56, 6*7 + 13*6 = 120, n*(n-1) + (2n-1)*(n-1) = (3n-1)*(n-1)
    )

    reward_range = (2 * REWARDS[False], 2 * REWARDS[True])  # final reward plus accumulated rewards
    observation_space = gymnasium.spaces.Graph(
        node_space=gymnasium.spaces.Discrete(3),  # empty, white, black
        edge_space=gymnasium.spaces.Discrete(4),  # none, white, black, opp color
    )
    action_space = gymnasium.spaces.Box(MINUS_ONE, ONE, (1,), float32)

    winner: Optional[bool]
    opening: str

    losses: DefaultDict = defaultdict(lambda: 0)

    def observation_from_board(self) -> Tuple[NDArray, NDArray, NDArray]:
        """"""

        # fmt: off
        return (
            *self.controller.board.to_graph(),
            BOARD_LINKS
        )
        # fmt: on


# print(HexBoard("a2c1b3a3", size=3).to_graph())
