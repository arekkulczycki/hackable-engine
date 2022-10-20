from typing import Tuple

import gym
import numpy

from arek_chess.common.constants import Print
from arek_chess.main.controller import Controller


class FullBoardEnv(gym.Env):
    # metadata = {}

    ACTION_SIZE = 7
    REWARDS = {
        '*': 0.0,
        '1/2-1/2': 0.0,
        '1-0': + 1.0,
        '0-1': - 1.0,
    }

    def __init__(self, fen: str):
        super().__init__()

        self.reward_range = (-1, 1)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.controller = Controller(Print.NOTHING, search_limit=10)
        self.controller.boot_up(fen, self.action_space.sample())

        self.obs = self.observation()

    def _get_action_space(self):
        return gym.spaces.Box(
            numpy.array([-1 for _ in range(self.ACTION_SIZE)], dtype=numpy.float32),
            numpy.array([1 for _ in range(self.ACTION_SIZE)], dtype=numpy.float32),
        )

    def _get_observation_space(self):
        return gym.spaces.MultiDiscrete([13 for _ in range(64)])

    def step(self, action: Tuple[numpy.float32, numpy.float32]):
        self._run_action(action)

        result = self.controller.board.result()
        if result == "*":
            self.controller.make_move()
            result = self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)

        return self.obs, reward, result != "*", {}

    def reset(self):
        self.render()
        self.controller.restart(self.action_space.sample())

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())

    def observation(self):
        return self._board_to_obs(self.controller.board.piece_map())

    def _board_to_obs(self, piece_map) -> list:
        observation = []
        for i in range(64):
            if i in piece_map:
                piece = piece_map[i]
                symbol = 1 if piece.symbol().islower() else -1
                piece_type = int(piece.piece_type)
                piece_int = piece_type * symbol + 6
                observation.append(piece_int)
            else:
                observation.append(0)
        return observation

    def _run_action(self, action: Tuple[numpy.float32, ...]) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
