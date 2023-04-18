import gym
import numpy

from arek_chess.common.constants import Print
from arek_chess.controller import Controller
from arek_chess.criteria.evaluation.base_eval import ActionType

DEFAULT_ACTION: ActionType = (
    numpy.double(100.0),
    numpy.double(1.0),
    numpy.double(-1.0),
    numpy.double(1.0),
    numpy.double(-1.0),
    numpy.double(5.0),
    numpy.double(10.0),
)


class SimpleEnv(gym.Env):
    # metadata = {}

    ACTION_SIZE = 7
    REWARDS = {
        "*": 0.0,
        "1/2-1/2": 0.0,
        "1-0": +1.0,
        "0-1": -1.0,
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
            numpy.array([-1 for _ in range(self.ACTION_SIZE)], dtype=numpy.double),
            numpy.array([1 for _ in range(self.ACTION_SIZE)], dtype=numpy.double),
        )

    def _get_observation_space(self):
        return gym.spaces.Box(numpy.array([0]), numpy.array([1]))

    def step(self, action: ActionType):
        self._run_action(tuple(action))

        result = self.controller.board.result()
        if result == "*":
            self.controller.make_move(DEFAULT_ACTION)
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

    def _board_to_obs(self) -> numpy.double:
        own_material = self.controller.board.get_material_simple(
            self.controller.board.turn
        )
        return numpy.double(own_material / 40)

    def _run_action(self, action: ActionType) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score
