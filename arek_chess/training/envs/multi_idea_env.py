from itertools import cycle
from time import sleep
from typing import Dict

import gym
import numpy
from numpy import float32

from arek_chess.common.constants import Print
from arek_chess.controller import Controller
from arek_chess.criteria.evaluation.base_eval import ActionType

DEFAULT_ACTION: ActionType = (
    double(10.0),  # is_check
    double(100.0),  # material
    double(5.0),  # mobility
    double(1.0),  # threats
    double(2.0),  # king threats
    double(2.0),  # direct threats
    double(-10.0),  # king safety/mobility
    double(3.0),  # protection
    double(0.0),  # light_pieces_white
    double(0.0),
    double(0.0),
    double(0.0),
    double(0.0),  # light_pawns_white
    double(0.0),
    double(0.0),
    double(0.0),
)
MEDIUM_ACTION: ActionType = (
    double(10.0),  # is_check
    double(25.0),  # material
    double(3.0),  # mobility
    double(2.0),  # threats
    double(2.0),  # king threats
    double(2.0),  # direct threats
    double(-3.0),  # king safety/mobility
    double(2.0),  # protection
    double(0.0),  # light_pieces_white
    double(0.0),
    double(0.0),
    double(0.0),
    double(0.0),  # light_pawns_white
    double(0.0),
    double(0.0),
    double(0.0),
)
WEAK_ACTION: ActionType = (
    double(10.0),  # is_check
    double(5.0),  # material
    double(0.0),  # mobility
    double(0.0),  # threats
    double(0.0),  # king threats
    double(0.0),  # direct threats
    double(0.0),  # king safety/mobility
    double(0.0),  # protection
    double(0.0),  # light_pieces_white
    double(0.0),
    double(0.0),
    double(0.0),
    double(0.0),  # light_pawns_white
    double(0.0),
    double(0.0),
    double(0.0),
)
ACTION_SIZE: int = 16
EQUAL_MIDDLEGAME_FEN = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
SHARP_MIDDLEGAME_FEN = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
ADVANTAGE_MIDDLEGAME_FEN = "r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9"
DISADVANTAGE_MIDDLEGAME_FEN = "rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8"
fens = cycle([EQUAL_MIDDLEGAME_FEN, SHARP_MIDDLEGAME_FEN, ADVANTAGE_MIDDLEGAME_FEN, DISADVANTAGE_MIDDLEGAME_FEN])


class MultiIdeaEnv(gym.Env):
    # metadata = {}

    REWARDS: Dict[str, double] = {
        "*": double(0.0),
        "1/2-1/2": double(0.0),
        "1-0": double(1.0),
        "0-1": double(-1.0),
    }

    def __init__(self):
        super().__init__()

        self.reward_range = (double(-1.0), double(1.0))

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.controller = Controller(Print.NOTHING, search_limit=9)
        self.controller.boot_up(next(fens), self.action_space.sample())

        self.obs = self.observation()
        self.steps_done = 0

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

    def step(self, action: ActionType):
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
            sleep(0.05)  # sleep is needed for the queues to clear, otherwise they crash...
            self.controller.make_move(DEFAULT_ACTION)

            result = self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)
        self.steps_done += 1

        return self.obs, reward, result != "*", {}

    def reset(self):
        self.render()
        self.controller.restart(fen=next(fens), action=self.action_space.sample())

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())
        print(self.steps_done)

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

    def _run_action(self, action: ActionType) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
