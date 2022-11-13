from itertools import cycle
from time import sleep
from typing import Dict, List

import gym
import numpy
from numpy import double

from arek_chess.common.constants import Print
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.main.controller import Controller

DEFAULT_ACTION: BaseEval.ActionType = (
    double(0.15),  # castling_rights
    double(-0.1),  # king_mobility
    double(0.1),  # is_check
    double(1.0),  # material
    double(0.015),  # own occupied square control
    double(0.015),  # opp occupied square control
    double(0.01),  # empty square control
    double(0.01),  # king proximity square control primary
    double(1.0),  # king proximity square control secondary
)
MEDIUM_ACTION: BaseEval.ActionType = (
    double(0.075),  # castling_rights
    double(-0.05),  # king_mobility
    double(0.1),  # is_check
    double(1.0),  # material
    double(0.1),  # occupied square control
    double(0.1),  # occupied square control
    double(0.0),  # empty square control
    double(0.0),  # king proximity square control primary
    double(0.0),  # king proximity square control secondary
)
WEAK_ACTION: BaseEval.ActionType = (
    double(0.0),  # castling_rights
    double(0.0),  # king_mobility
    double(0.1),  # is_check
    double(1.0),  # material
    double(0.0),  # occupied square control
    double(0.0),  # occupied square control
    double(0.0),  # empty square control
    double(0.0),  # king proximity square control primary
    double(0.0),  # king proximity square control secondary
)
ACTION_SIZE: int = 9
# EQUAL_MIDDLEGAME_FEN = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
# SHARP_MIDDLEGAME_FEN = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
# ADVANTAGE_MIDDLEGAME_FEN = "r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9"
# DISADVANTAGE_MIDDLEGAME_FEN = "rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8"
# fens = cycle([EQUAL_MIDDLEGAME_FEN, SHARP_MIDDLEGAME_FEN, ADVANTAGE_MIDDLEGAME_FEN, DISADVANTAGE_MIDDLEGAME_FEN])


class SquareControlEnv(gym.Env):
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
        # self.controller.boot_up(next(fens), self.action_space.sample())
        self.controller.boot_up(action=self.action_space.sample())

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

        [x] own king mobility (king safety)
        [x] opponents king mobility (king safety)
        [x] material on board
        [x] pawns on board (how open is the position)
        """

        return gym.spaces.Box(
            numpy.array([0 for _ in range(4)], dtype=numpy.double),
            numpy.array([1 for _ in range(4)], dtype=numpy.double),
        )

    def step(self, action: BaseEval.ActionType):
        # artificially multiply material value for easier random discovery
        # artificially allow negative king-mobility value
        action = tuple(action)
        self._run_action(action)

        result = self.controller.board.result()
        if result == "*":
            # playing against random action
            # self.controller.make_move(self.action_space.sample())

            # playing against configured action
            sleep(0.05)  # sleep is needed for the queues to clear, otherwise they crash...
            self.controller.make_move(WEAK_ACTION)

            result = self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)
        self.steps_done += 1

        return self.obs, reward, result != "*", {}

    def reset(self):
        self.render()
        # self.controller.restart(fen=next(fens), action=self.action_space.sample())
        self.controller.restart(action=self.action_space.sample())

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())
        print(self.steps_done)

    def observation(self):
        return self._board_to_obs()

    def _board_to_obs(self) -> List[double]:
        board = self.controller.board
        own_king_mobility = double(board.get_king_mobility(board.turn) / 8.0)
        opp_king_mobility = double(board.get_king_mobility(not board.turn) / 8.0)
        material = double(board.get_material_simple_both() / 40.0)
        pawns = double(board.get_pawns_simple_both() / 16.0)

        return [own_king_mobility, opp_king_mobility, material, pawns]

    def _run_action(self, action: BaseEval.ActionType) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
