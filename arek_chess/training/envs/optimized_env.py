from typing import Dict

import gym
import numpy
from numpy import double

from arek_chess.common.constants import Print
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.main.controller import Controller

DEFAULT_ACTION: BaseEval.ActionType = (
    double(10.0),  # is_check
    double(100.0),  # material
    double(5.0),  # mobility
    double(1.0),  # threats
    double(2.0),  # king threats
    double(2.0),  # direct threats
    double(-10.0),  # king safety/mobility
    double(0.0),  # light_pieces_white
    double(0.0),
    double(0.0),
    double(0.0),
    double(0.0),  # light_pawns_white
    double(0.0),
    double(0.0),
    double(0.0),
)
MEDIUM_ACTION: BaseEval.ActionType = (
    double(10.0),  # is_check
    double(25.0),  # material
    double(3.0),  # mobility
    double(2.0),  # threats
    double(2.0),  # king threats
    double(2.0),  # direct threats
    double(-3.0),  # king safety/mobility
    double(0.0),  # light_pieces_white
    double(0.0),
    double(0.0),
    double(0.0),
    double(0.0),  # light_pawns_white
    double(0.0),
    double(0.0),
    double(0.0),
)
ACTION_SIZE: int = 15


class OptimizedEnv(gym.Env):
    # metadata = {}

    REWARDS: Dict[str, double] = {
        "*": double(0.0),
        "1/2-1/2": double(0.0),
        "1-0": double(1.0),
        "0-1": double(-1.0),
    }

    def __init__(self, fen: str):
        super().__init__()

        self.reward_range = (double(-1.0), double(1.0))

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.controller = Controller(Print.NOTHING, search_limit=9)
        self.controller.boot_up(fen, self.action_space.sample())

        self.obs = self.observation()

    def _get_action_space(self):
        return gym.spaces.Box(
            numpy.array([0 for _ in range(ACTION_SIZE)], dtype=double),
            numpy.array([1 for _ in range(ACTION_SIZE)], dtype=double),
        )

    def _get_observation_space(self):
        """
        Optimized state of the board.

        material: normalized [0, 1]
        white king: discrete 1-64
        black king: discrete 1-64
        white threats: List[double] of length 64
        black threats: List[double] of length 64
        white bishop: bool
        black bishop: bool
        """

        spaces = {
            "material": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=double),
            "white_threats": gym.spaces.Box(low=0, high=1, shape=(64,), dtype=double),
            "black_threats": gym.spaces.Box(low=0, high=1, shape=(64,), dtype=double),
            "white_king": gym.spaces.Discrete(64),
            "black_king": gym.spaces.Discrete(64),
            "own_light_bishop": gym.spaces.Discrete(2),
            "own_dark_bishop": gym.spaces.Discrete(2),
            "opp_light_bishop": gym.spaces.Discrete(2),
            "opp_dark_bishop": gym.spaces.Discrete(2),
        }

        return gym.spaces.Dict(spaces)

    def step(self, action: BaseEval.ActionType):
        # artificially multiply material value for easier random discovery
        # artificially allow negative king-mobility value
        action = tuple(
            v * 10 if i == 1 else 2 * (v - 0.5) if i == 6 else v
            for i, v in enumerate(action)
        )
        self._run_action(action)

        result = self.controller.board.result()
        if result == "*":
            # playing against random action
            # self.controller.make_move(self.action_space.sample())

            # playing against configured action
            self.controller.make_move(MEDIUM_ACTION)

            result = self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)

        return self.obs, reward, result != "*", {}

    def reset(self):
        self.render()
        self.controller.restart(action=self.action_space.sample())

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())

    def observation(self):
        return self._board_to_obs()

    def _board_to_obs(self) -> Dict[str, double]:
        board = self.controller.board
        material = double(board.get_material_simple_both() / 40)
        white_threats = board.get_normalized_threats_map(True)
        black_threats = board.get_normalized_threats_map(False)
        white_king = board.king(True)
        black_king = board.king(False)
        own_light_bishop = board.has_white_bishop(True)  # TODO: take as arg which side the engine is playing
        own_dark_bishop = board.has_black_bishop(True)
        opp_light_bishop = board.has_white_bishop(False)
        opp_dark_bishop = board.has_black_bishop(False)

        return {
            "material": material,
            "white_threats": white_threats,
            "black_threats": black_threats,
            "white_king": white_king,
            "black_king": black_king,
            "own_light_bishop": own_light_bishop,
            "own_dark_bishop": own_dark_bishop,
            "opp_light_bishop": opp_light_bishop,
            "opp_dark_bishop": opp_dark_bishop,
        }

    def _run_action(self, action: BaseEval.ActionType) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
