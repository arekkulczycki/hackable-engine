from typing import Tuple

import gym
import numpy

from arek_chess.main.controller import Controller


class BaseEnv(gym.Env):
    # metadata = {}

    ACTION_SIZE = 7
    REWARDS = {
        '*': 0.0,
        '1/2-1/2': 0.0,
        '1-0': + 1.0,
        '0-1': - 1.0,
    }

    def __init__(self, initial_fen: str):
        super().__init__()

        self.reward_range = (-1, 1)

        self.action_space = gym.spaces.Box(
            numpy.array([0 for _ in range(self.ACTION_SIZE)], dtype=numpy.float32),
            numpy.array([1 for _ in range(self.ACTION_SIZE)], dtype=numpy.float32),
        )
        self.observation_space = gym.spaces.MultiDiscrete([13 for _ in range(64)])

        self.controller = Controller()
        self.controller.boot_up(initial_fen)

        self.obs = self.observation()

    def step(self, action: Tuple[numpy.float32, numpy.float32]):
        self._run_action(action)

        self.obs = self.observation()
        reward = self._get_reward()

        return self.obs, reward, self.controller.board.is_game_over(), {}

    def reset(self):
        self.controller.restart()

        return self.observation()

    def render(self, mode="human", close=False):
        print(self.obs)

    def observation(self):
        return self.board_to_obs(self.controller.board)

    def board_to_obs(self, piece_map) -> list:
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

    def _get_reward(self):
        return self.REWARDS[self.controller.board.result()]

    def _get_intermediate_reward(self):
        return # maybe some material-based score
