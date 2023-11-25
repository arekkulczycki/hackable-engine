# -*- coding: utf-8 -*-
import math
from collections import defaultdict
from itertools import cycle
from random import choice
from typing import Any, DefaultDict, Dict, Generator, List, Optional, SupportsFloat, Tuple

import gym
import onnxruntime as ort
from gymnasium.core import ActType, ObsType, RenderFrame
from nptyping import Int8, NDArray, Shape
from numpy import asarray, float32, int8, eye

from arek_chess.board.hex.hex_board import HexBoard, Move
from arek_chess.common.constants import INF
from arek_chess.controller import Controller

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
        # "a4",
        # "a5",
        # "e1",
        # "e2",
        "a1",
        "a2",
        "a3",
        "e3",
        "e4",
        "e5",
        "d1",
        "b5",
        "a1c3a4",
        "a2c3a4",
        "a3b4d3",
        "e3d2b3",
        "e4c3e2",
        "e5c3e2",
        "d1b4d3",
        "d1c4b4",
        "d1c4b4c2d2",
        "b5d2b3",
        "b5c2d2",
        "b5c2d2c4b4",
    ]
)


# class SimpleEnv(gymnasium.Env):
class Raw5x5BinEnv(gym.Env):
    """"""

    BOARD_SIZE: int = 5
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.MultiBinary(BOARD_SIZE ** 2 * 2)
    action_space = gym.spaces.Box(MINUS_ONE, ONE, (1,), float32)

    winner: Optional[bool]
    opening: str

    losses: DefaultDict = defaultdict(lambda: 0)

    def __init__(self, *, controller: Optional[Controller] = None):
        super().__init__()

        if controller:
            self._setup_controller(controller)
            self.obs = self.observation_from_board()

        self.steps_done = 0
        self.winner = None

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        self.best_move: Optional[Tuple[Move, Tuple[float32]]] = None
        self.current_move: Optional[Move] = None

        # self.opp_model = onnx.load("raw7hex4.onnx")
        # self.opp_model = ort.InferenceSession(
        #     "raw7hex.onnx", providers=["AzureExecutionProvider", "CPUExecutionProvider"]
        # )

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.opening = next(openings)
        self.controller._setup_board(self.opening, size=self.BOARD_SIZE)
        self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        notation = self.controller.board.get_notation()
        print("render: ", notation, self._get_reward(self.winner))
        print("steps done: ", self.steps_done)
        return notation

    def reset(
        self,
        *,
        # seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # super().reset(seed=seed)
        if not self.winner:
            self.losses[self.opening] += 1

        self.render()
        self.opening = next(openings)
        self.controller.reset_board(self.opening, size=self.BOARD_SIZE)

        self._prepare_child_moves()

        # return self.observation_from_board(), {}
        return self.observation_from_board()

    def _prepare_child_moves(self) -> None:
        """
        Reset generator and play the first option to be evaluated.
        """

        self.moves = self.controller.board.legal_moves  # returns new generator
        self.current_move = next(self.moves)
        self.best_move = None
        self.controller.board.push(self.current_move)

    def step(
        self,
        action: ActType
        # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    ) -> Tuple[ObsType, SupportsFloat, bool, Dict[str, Any]]:
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        if self.best_move is None:
            self.best_move = (self.current_move, action[0])
        else:
            best_score = self.best_move[1]
            current_score = action[0]
            if current_score > best_score:
                self.best_move = (self.current_move, action[0])

        try:
            # if doesn't raise then there are still moves to be evaluated
            self.current_move = next(self.moves)

            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated in the next `step`
            self.controller.board.push(self.current_move)

            winner = None

        except StopIteration:
            # all moves evaluated - undo the last and push the best move on the board
            # print("best move: ", self.best_move[0].get_coord(), self.best_move[1])
            self.controller.board.pop()
            self.controller.board.push(self.best_move[0])

            winner = self._get_winner()

            # if the last move didn't conclude the game, now play opponent move and reset generator
            if winner is None:
                # playing against a configured action or an action from self-trained model
                # obs = reshape(self.observation_from_board().astype(float32), (1, 49))
                # opp_action, value = self.opp_model.run(None, {"input": obs})
                # self.controller.make_move(self.prepare_action(opp_action[0]))

                self.controller.make_move(DEFAULT_ACTION)
                # self._make_random_move()

                winner = self._get_winner()

                # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
                if winner is None:
                    self._prepare_child_moves()

        self.obs = self.observation_from_board()

        self.winner = winner
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
        if winner is False:
            # - 1 + ((len(self.controller.board.move_stack) + 1) - 12) / (25 - 12)
            # return -1 + max(((len(self.controller.board.move_stack) - 9) / 16, 0))
            return -1 + max(((len(self.controller.board.move_stack) - 7) / 14, 0))
        elif winner is True:
            # return 1 - max((((len(self.controller.board.move_stack) - 9) / 16) ** 2, 0))
            return 1 - max(((len(self.controller.board.move_stack) - 7) / 14, 0))

        return self.REWARDS[winner]

    def _make_random_move(self):
        moves = list(self.controller.board.legal_moves)
        self.controller.board.push(choice(moves))

    def observation_from_board(self) -> NDArray[Shape["50"], Int8]:
        local: NDArray[Shape["25"], Int8] = self.controller.board.get_neighbourhood(
            5, should_suppress=True
        )
        # fmt: off
        return eye(3, dtype=int8)[local][:, 1:].flatten()  # dummy encoding - 2 columns of 0/1 values, 1 column dropped
        # fmt: on

    @staticmethod
    def prepare_action(action):
        return asarray((0, 0, 0, 0, 0, 10, action[0], 1))

    def summarize(self):
        print("loss summary: ", sorted(self.losses.items(), key=lambda x: -x[1]))
