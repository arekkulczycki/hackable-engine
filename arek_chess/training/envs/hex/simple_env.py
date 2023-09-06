# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gym
from gymnasium.core import ActType, ObsType, RenderFrame
from nptyping import Int8, NDArray, Shape
from numpy import asarray, float32, int8

from arek_chess.board.hex.hex_board import HexBoard
from arek_chess.common.constants import INF
from arek_chess.controller import Controller

BOARD_SIZE: int = 7
DEFAULT_ACTION = asarray((
    float32(1.0),
    float32(1.0),
    float32(4.0),
    float32(1.0),
    float32(10.0),  # missing distance to a complete connection
    float32(10.0),  # turn bonus
    float32(0.0),  # local pattern eval
    float32(0.0),  # local pattern confidence
))
ACTION_SIZE: int = 8

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

openings = cycle(
    [
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
    ]
)


# class SimpleEnv(gymnasium.Env):
class SimpleEnv(gym.Env):
    """"""

    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.Dict(
        {
            "global": gym.spaces.Box(ZERO, ONE, shape=(9,), dtype=float32),
            "local": gym.spaces.MultiDiscrete(
                asarray([3 for _ in range(49)]), dtype=int8
            ),
            # gym.spaces.Box(
            #     ZERO, THREE, shape=(49,), dtype=int8
            # ),
            # gymnasium.spaces.MultiDiscrete(
            #     asarray([3 for _ in range(49)]), dtype=int8, start=[int8(0) for _ in range(49)]
            # ),
        }
    )
    # action_space = gymnasium.spaces.Box(ZERO, ONE, (6,), float32)
    action_space = gym.spaces.Box(ZERO, ONE, (ACTION_SIZE,), float32)

    winner: Optional[bool]

    def __init__(self, *, controller: Optional[Controller] = None):
        super().__init__()

        if controller:
            self._setup_controller(controller)
            self.obs = self.observation_from_board()

        self.steps_done = 0
        self.winner = None

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(openings))
        self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        notation = self.controller.board.get_notation()
        print(notation, self.winner)
        print(self.steps_done)
        return notation

    def reset(
        self,
        *,
        # seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # super().reset(seed=seed)

        self.render()
        self.controller.reset_board(next(openings))

        # return self.observation_from_board(), {}
        return self.observation_from_board()

    def step(
        self, action: ActType
    # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    ) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        self.controller.make_move(action)

        winner = self._get_winner()

        if winner is None:
            # playing against a configured action
            self.controller.make_move(DEFAULT_ACTION)

            winner = self._get_winner()

        self.obs = self.observation_from_board()
        reward = self._get_reward(winner)
        self.steps_done += 1

        # return self.obs, reward, winner is not None, False, {}
        return self.obs, reward, winner is not None, {}

    def _get_winner(self) -> Optional[bool]:
        winner = self.controller.board.winner()
        self.winner = winner

        if winner is None:
            if abs(self.controller.search_worker.root.score) == INF:
                return self.controller.search_worker.root.score > 0

        return winner

    def _get_reward(self, winner):
        return self.REWARDS[winner]

    def observation_from_board(self):
        (
            connectedness_white,
            connectedness_black,
            wingspan_white,
            wingspan_black,
        ) = self.controller.board.get_connectedness_and_wingspan()
        balance_white, centrishness_white = self.controller.board.get_imbalance(True)
        balance_black, centrishness_black = self.controller.board.get_imbalance(False)
        n_moves = len(self.controller.board.move_stack)
        local: NDArray[Shape["49"], Int8] = self.controller.board.get_neighbourhood()

        return {
            "global": asarray((
                connectedness_white,
                connectedness_black,
                wingspan_white,
                wingspan_black,
                balance_white,
                balance_black,
                centrishness_white,
                centrishness_black,
                n_moves,
            ), dtype=float32),
            "local": local,
        }
