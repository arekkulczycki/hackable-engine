# -*- coding: utf-8 -*-
import sys
from itertools import cycle
from typing import Dict, Optional

import gym
import numpy
from numpy import float32

from hackable_engine.controller import Controller
from hackable_engine.criteria.evaluation.base_eval import WeightsType
from hackable_engine.training.envs.chess.square_control_env_single_action_util import _board_to_obs

DEFAULT_ACTION: WeightsType = (
    float32(-0.05),  # king_mobility
    float32(0.05),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.0125),  # own occupied square control
    float32(0.0125),  # opp occupied square control
    float32(0.01),  # empty square control
    # float32(0.015),  # empty square control nominal
    float32(0.015),  # own king proximity square control
    float32(0.01),  # opp king proximity square control
    float32(0.15),  # turn
)
MEDIUM_ACTION: WeightsType = (
    float32(-0.05),  # king_mobility
    float32(0.1),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.1),  # own occupied square control
    float32(0.1),  # opp occupied square control
    float32(0.0),  # empty square control
    float32(0.0),  # own king proximity square control
    float32(0.0),  # opp king proximity square control
    float32(0.1),  # turn
)
WEAK_ACTION: WeightsType = (
    float32(0.0),  # king_mobility
    float32(0.0),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.0),  # own occupied square control
    float32(0.0),  # opp occupied square control
    float32(0.0),  # empty square control
    float32(0.0),  # own king proximity square control
    float32(0.0),  # opp king proximity square control
    float32(0.1),  # turn
)
PARAMS_NUMBER: int = 10
INITIAL_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
EQUAL_MIDDLEGAME_FEN = (
    "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
)
SHARP_MIDDLEGAME_FEN = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
ADVANTAGE_MIDDLEGAME_FEN = (
    "r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9"
)
DISADVANTAGE_MIDDLEGAME_FEN = (
    "rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8"
)
fens = cycle(
    [
        EQUAL_MIDDLEGAME_FEN,
        SHARP_MIDDLEGAME_FEN,
        ADVANTAGE_MIDDLEGAME_FEN,
        DISADVANTAGE_MIDDLEGAME_FEN,
    ]
)
# fens = cycle([INITIAL_FEN])


class SquareControlEnv(gym.Env):
    # metadata = {}

    REWARDS: Dict[str, float32] = {
        "*": float32(0.0),
        "1/2-1/2": float32(0.0),
        "1-0": float32(1.0),
        "0-1": float32(-1.0),
    }

    def __init__(self, *, controller: Optional[Controller] = None):
        super().__init__()

        self.reward_range = (float32(-1.0), float32(1.0))

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self._set_controller(controller)

        self.obs = self.observation()
        self.steps_done = 0

    def _set_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(fens))
        self.controller.boot_up()

    def _get_action_space(self):
        return gym.spaces.Box(-1, 1, (1,))  # float32 is the default

    def _get_observation_space(self):
        """
        Optimized state of the board.

        [x] own king mobility (king safety)
        [x] opponents king mobility (king safety)
        [x] material on board
        [x] pawns on board (how open is the position)
        [x] space of each player
        [x] king proximity
        [ ] bishops on board (color)
        """

        return gym.spaces.Tuple(
            [
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(2),
                gym.spaces.Discrete(2),
                gym.spaces.Box(
                    numpy.array(
                        [-1 if i <= 1 else 0 for i in range(12)], dtype=numpy.float32
                    ),
                    numpy.array([1 for _ in range(12)], dtype=numpy.float32),
                ),
            ]
        )

    def step(self, action: WeightsType):
        action = self.action_upgrade(action)
        self._run_action(action)

        result = self.get_result()  # self.controller.board.result()
        if result == "*":
            # playing against a configured action
            self._run_action(WEAK_ACTION)

            result = self.get_result()  # self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)
        self.steps_done += 1

        return self.obs, reward, result != "*", {}

    def get_result(self):
        result = self.controller.board.result()
        if result == "*":
            white_material = self.controller.board.get_material_simple(True)
            black_material = self.controller.board.get_material_simple(False)
            dif = white_material - black_material
            if dif > 12:
                return "1-0"
            elif dif < -12:
                return "0-1"
        return result

    def reset(self):
        self.render()
        self.controller.reset(next(fens))

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())
        print(self.steps_done)

    def observation(self):
        return _board_to_obs(self.controller.board)

    def _run_action(self, action: WeightsType) -> None:
        try:
            self.controller.make_move(action)
        except RuntimeError:
            sys.exit()
            # self._set_controller()
            # self._run_action(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score

    @staticmethod
    def action_upgrade(action: WeightsType) -> WeightsType:
        return action

        # artificially make turn value smaller
        # artificially multiply material value for easier random discovery
        # return tuple(
        #    v / 2 if i == 9 else v * 2 if i == 3 else v for i, v in enumerate(action)
        # )

    @staticmethod
    def action_downgrade(action: WeightsType) -> WeightsType:
        return action
        # return tuple(
        #     (v / 2 + 0.5) if i == 1 else v / 2 if i == 3 else v
        #     for i, v in enumerate(action)
        # )
