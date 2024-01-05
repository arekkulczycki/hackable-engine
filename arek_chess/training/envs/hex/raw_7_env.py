# -*- coding: utf-8 -*-
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, Optional, SupportsFloat, Tuple

import gym
import onnxruntime as ort
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from nptyping import Int8, NDArray, Shape
from numpy import asarray, float32, reshape, full, int8
from bfloat16 import bfloat16

from arek_chess.board.hex.hex_board import HexBoard, Move
from arek_chess.common.constants import Game, INF, Print
from arek_chess.controller import Controller

LESS_THAN_ZERO: float32 = float32(-0.00001)
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
        "g1",
        "g2",
        "g3",
        "g4",
        "g5",
        "g6",
        "g7",
        "a1c5f5e3d4",
        "a1d4c5c4e4e3g2",
        "a1e3d4d3b4",
        "a1c3e3e4d4",
        "a1c3d4e2f2",
        "g7e5d4c6b6",
        "g7e5c5c4d4",
        "g7c5d4d5f4",
        "g7d4e3e4c4c5a6",
        "g7c5b3e3d4",
        # same openings twice for counterbalance against single move opening
        "a1c5f5e3d4",
        "a1d4c5c4e4e3g2",
        "a1e3d4d3b4",
        "a1c3e3e4d4",
        "a1c3d4e2f2",
        "g7e5d4c6b6",
        "g7e5c5c4d4",
        "g7c5d4d5f4",
        "g7d4e3e4c4c5a6",
        "g7c5b3e3d4",
    ]
)


class Raw7Env(gym.Env):
    """"""

    BOARD_SIZE: int = 7
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.Box(0, 2, shape=(BOARD_SIZE, BOARD_SIZE), dtype=float32)  # should be int8
    action_space = gym.spaces.Box(ZERO, ONE, shape=(1,), dtype=float32)

    winner: Optional[bool]

    def __init__(self, *args, controller: Optional[Controller] = None):
        super().__init__()

        if controller:
            self._setup_controller(controller)
            self.obs = self.observation_from_board()
        else:
            self._setup_controller(Controller(
                printing=Print.NOTHING,
                # tree_params="4,4,",
                search_limit=9,
                is_training_run=True,
                in_thread=False,
                timeout=3,
                game=Game.HEX,
                board_size=self.BOARD_SIZE,
            ))

        self.steps_done = 0
        self.winner = None

        self.losses = 0

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        self.best_move: Optional[Tuple[Move, Tuple[float32]]] = None
        self.current_move: Optional[Move] = None

        # self.opp_model = onnx.load("onnx_hex7")
        # self.opp_model = ort.InferenceSession(
        #     "onnx_hex7", providers=["CPUExecutionProvider"]
        # )

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(openings), size=self.BOARD_SIZE)
        # don't boot up if theres no need for engine to run, but to train on multiple processes
        # self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        notation = self.controller.board.get_notation()
        print("render: ", notation, self._get_reward(self.winner))
        print("steps done: ", self.steps_done)
        return notation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)

        self.render()

        notation = next(openings)
        self.controller.reset_board(notation, size=self.BOARD_SIZE, init_move_stack=True)

        obs = self.observation_from_board()

        # must be after the observation, as it adds a move on the board
        self._prepare_child_moves()
        return obs, {}

    def _prepare_child_moves(self) -> None:
        """
        Reset generator and play the first option to be evaluated.
        """

        self.moves = self.controller.board.legal_moves  # returns new generator
        self.current_move = next(self.moves)
        self.best_move = None
        self.controller.board.push(self.current_move)

    def step_iterative(
        self,
        action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
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

        except StopIteration:
            # all moves evaluated - undo the last and push the best move on the board
            # print("best move: ", self.best_move[0].get_coord(), self.best_move[1])
            self.controller.board.pop()
            self.controller.board.push(self.best_move[0])

            winner = self._get_winner()

            # if the last move didn't conclude the game, now play opponent move and reset generator
            if winner is None:
                # self.controller.make_move(self.prepare_action(opp_action[0]))
                # self.controller.make_move(DEFAULT_ACTION, search_limit=choice((0, 8)))

                # self._make_self_trained_move(False)  # opponent playing black
                self._make_random_move()

                winner = self._get_winner()

                # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
                if winner is None:
                    self._prepare_child_moves()

        else:
            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated in the next `step`
            self.controller.board.push(self.current_move)

            winner = None

        self.obs = self.observation_from_board()

        self.winner = winner
        reward = self._get_reward(winner, action[0])
        self.steps_done += 1

        # return self.obs, reward, winner is not None, False, {}
        return self.obs, reward, winner is not None, False, {}

    def step(
        self,
        action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """"""

        return self.step_iterative(action)

        # self.controller.board.push(self.get_move_from_action())
        #
        # winner = self._get_winner()
        # if winner is None:
        #     self._make_random_move()
        #
        # self.obs = self.observation_from_board()
        #
        # self.winner = winner
        # reward = self._get_reward(winner)
        # self.steps_done += 1
        #
        # # return self.obs, reward, winner is not None, False, {}
        # return self.obs, reward, winner is not None, {}

    def _get_winner(self) -> Optional[bool]:
        winner = self.controller.board.winner()
        self.winner = winner

        if winner is None:
            if abs(self.controller.search_worker.root.score) == INF:
                return self.controller.search_worker.root.score > 0

        return winner

    def _get_reward(self, winner, score=None):
        if winner is False:
            # - 1 + ((len(self.controller.board.move_stack) + 1) - 12) / (25 - 12)
            return -1 + max(0, (len(self.controller.board.move_stack) - 12)) / 36

        elif winner is True:
            return 1 - (max(0, (len(self.controller.board.move_stack) - 12)) / 36) ** 0.5

        # return ZERO if score != ZERO and score != ONE else LESS_THAN_ZERO
        return 0  # if score != ZERO and score != ONE else -0.003

    def _make_random_move(self):
        moves = list(self.controller.board.legal_moves)
        self.controller.board.push(choice(moves))

    def _make_self_trained_move(self, color: bool) -> None:
        best_move: Optional[Move] = None
        best_score = None
        for move in self.controller.board.legal_moves:
            self.controller.board.push(move)
            obs = reshape(self.observation_from_board().numpy().astype(float32), (1, 98))
            score, value = self.opp_model.run(None, {"input": obs})
            if best_move is None or (color and score > best_score) or (not color and score < best_score):
                best_move = move
                best_score = score
            self.controller.board.pop()

        self.controller.board.push(best_move)

    def observation_from_board(self) -> th.Tensor:
        return th.from_numpy(self.controller.board.get_neighbourhood(
            7, should_suppress=True
        ))  # .reshape(1, 1, self.BOARD_SIZE, self.BOARD_SIZE))

    @staticmethod
    def prepare_action(action):
        return asarray((0, 0, 0, 0, 0, 10, action[0], 1))

    def summarize(self):
        print("loss summary: ", self.losses)
        # print("loss summary: ", sorted(self.losses.items(), key=lambda x: -x[1]))
