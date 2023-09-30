# -*- coding: utf-8 -*-
import math
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, List, Optional, SupportsFloat, Tuple

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
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "a9",
        "i1",
        "i2",
        "i3",
        "i4",
        "i5",
        "i6",
        "i7",
        "i8",
        "i9",
        "c2",
        "g8",
        "g2",
        "c8",
        ### middlegame
        "c2d6g6g3b5",
        "g8f4c4c7h5",
        "a9d3b7f7f4e5c4",
        "i1f7h3d3d6e5g6",
        "a9d3b7f7f4e5f6",
        "i1f7h3d3d6e5d4",
        "a7b8g6g7h6h7f7f4b3",
        "i3h2c4c3b4b3d3d6h7",
        ### endgame
        "c2d6g6g3b5f7g7f8g8",
        "g8f4c4c7h5d3c3d2c2",
        "a9d3b7f7f4e5f6f5g6h2g3",
        "i1f7h3d3d6e5d4d5c4b8c7",
        "a9d3b7f7f4e5c4c5b4d4f6",
        "i1f7h3d3d6e5g6g5h6f6d4",
        "a7b8g6g7h6h7f7f4b3c3b4c2b2c4b5c5",
        "i3h2c4c3b4b3d3d6h7g7h6g8h8g6h5g5",
        "a7b8g6g7h6h7f7f4b3e3b5",
        "i3h2c4c3b4b3d3d6h7e7h5",
        ### finishing touch
        "c2d6g6g3b5f7g7f8g8f6g4h3g5e4b7",
        "c2d6g6g3b5f7g7f8g8f6g5e4f4f3b7",
        "g8f4c4c7h5d3c3d2c2d4c5e6d6d7h3",
        "g8f4c4c7h5d3c3d2c2d4c6b7c5e6h3",
        "a9d3b7f7f4e5c4c3b4b3e4d5f6f5h4g5i5",
        "i1f7h3d3d6e5g6g7h6h7e6f5d4d5b6c5a5",
        "a9d3b7f7f4e5f6f5g6h2g3h4c4c3b4b3d4",
        "i1f7h3d3d6e5g6g7h6h7d4d5c4b8c7b6f6",
        "a7b8g6g7h6h7f7f4b3c3b4c2b2c4b5c5h3f5h4h2g3g2f3f2e3e2b7",
        "i3h2c4c3b4b3d3d6h7g7h6g8h8g6h5g5b7d5b6b8c7c8d7d8e7e8h3",
        "a7b8g6g7h6h7f7f4b3f3d5e6f5e5h2h3d7",
        "i3h2c4c3b4b3d3d6h7d7f5e4d5e5b8b7f3",
        "a7b8g6g7h6h7f7f4b3f3c7c8d7d8d9",
        "i3h2c4c3b4b3d3d6h7d7g3g2f3f2f1",
    ]
)


# class SimpleEnv(gymnasium.Env):
class Raw9x9Env(gym.Env):
    """"""

    BOARD_SIZE: int = 9
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.MultiBinary(BOARD_SIZE ** 2 * 2)
    action_space = gym.spaces.Box(ZERO, ONE, (1,), float32)

    winner: Optional[bool]

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
        self.controller._setup_board(next(openings), size=self.BOARD_SIZE)
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

        self.render()
        self.controller.reset_board(next(openings), size=self.BOARD_SIZE)

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

                # self.controller.make_move(DEFAULT_ACTION, search_limit=choice((0, 8)))

                self._make_random_move()

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
            return -1 + (len(self.controller.board.move_stack) - 16) / (81-16)

        elif winner is True:
            return 1 - ((len(self.controller.board.move_stack) - 17) / (81-17)) ** 2  # squared for "optimism"

        return self.REWARDS[winner]

    def _make_random_move(self):
        moves = list(self.controller.board.legal_moves)
        self.controller.board.push(choice(moves))

    def observation_from_board(self) -> NDArray[Shape["162"], Int8]:
        local: NDArray[Shape["81"], Int8] = self.controller.board.get_neighbourhood(
            9, should_suppress=True
        )
        # fmt: off
        return eye(3, dtype=int8)[local][:, 1:].flatten()  # dummy encoding - 2 columns of 0/1 values, 1 column dropped
        # fmt: on

    @staticmethod
    def prepare_action(action):
        return asarray((0, 0, 0, 0, 0, 10, action[0], 1))
