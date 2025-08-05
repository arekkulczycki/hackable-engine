from abc import ABC
from typing import Tuple

import gym
import numpy

from hackable_engine.controller import Controller


class BaseEnv(gym.Env, ABC):
    # metadata = {}

    PARAMS_NUMBER = 7
    REWARDS = {
        '*': 0.0,
        '1/2-1/2': 0.0,
        '1-0': + 1.0,
        '0-1': - 1.0,
    }

    def __init__(self):
        super().__init__()

        self.reward_range = (-1, 1)

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        self.controller = Controller()
        self.controller.boot_up()

        self.obs = self.observation()

    def _get_action_space(self):
        raise NotImplemented

    def _get_observation_space(self):
        raise NotImplemented

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
        raise NotImplemented

    def _run_action(self, action: Tuple[numpy.float32, ...]) -> None:
        self.controller.make_move(action)

    def _get_reward(self):
        return self.REWARDS[self.controller.board.result()]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
