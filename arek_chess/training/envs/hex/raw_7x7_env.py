# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Any, Dict, Optional, SupportsFloat, Tuple

import gym
import onnxruntime as ort
from gymnasium.core import ActType, ObsType, RenderFrame
from nptyping import Int8, NDArray, Shape
from numpy import asarray, float32, int8

from arek_chess.common.constants import INF
from arek_chess.controller import Controller

BOARD_SIZE: int = 7
DEFAULT_ACTION = asarray(
    (
        float32(1.0),
        float32(1.0),
        float32(5.0),
        float32(1.0),
        float32(15.0),
        float32(15.0),
        float32(0.0),  # local pattern eval
        float32(0.0),  # local pattern confidence
    )
)
ACTION_SIZE: int = 8

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

openings = cycle(
    [
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        # "a8",
        # "a9",
    ]
)


# class SimpleEnv(gymnasium.Env):
class Raw7x7Env(gym.Env):
    """"""

    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.MultiDiscrete(
        asarray([3 for _ in range(49)], dtype=int8), dtype=int8
    )
    action_space = gym.spaces.Box(ZERO, ONE, (1,), float32)

    winner: Optional[bool]

    def __init__(self, *, controller: Optional[Controller] = None):
        super().__init__()

        if controller:
            self._setup_controller(controller)
            self.obs = self.observation_from_board()

        self.steps_done = 0
        self.winner = None

        # self.opp_model = onnx.load("raw7hex4.onnx")
        # self.opp_model = ort.InferenceSession(
        #     "raw7hex.onnx", providers=["AzureExecutionProvider", "CPUExecutionProvider"]
        # )

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(openings), size=BOARD_SIZE)
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
        self.controller.reset_board(next(openings), size=BOARD_SIZE)

        # return self.observation_from_board(), {}
        return self.observation_from_board()

    def step(
        self,
        action: ActType
        # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    ) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        self.controller.make_move(self.prepare_action(action), search_limit=0)

        winner = self._get_winner()

        if winner is None:
            # playing against a configured action or an action from self-trained model
            # obs = reshape(self.observation_from_board().astype(float32), (1, 49))
            # opp_action, value = self.opp_model.run(None, {"input": obs})
            # self.controller.make_move(self.prepare_action(opp_action[0]))
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

    # def observation_from_board(self):
    #     local: NDArray[Shape["49"], Int8] = self.controller.board.get_neighbourhood(
    #         7, should_suppress=True
    #     )
    #
    #     return local

    def observation_from_board(self):
        local: NDArray[Shape["49"], Int8] = self.controller.board.get_neighbourhood(
            7, should_suppress=True
        )


        return local

    def prepare_action(self, action):
        return asarray((0, 0, 0, 0, 0, 10, action[0], 1))
