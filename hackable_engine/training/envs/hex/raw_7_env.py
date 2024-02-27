# -*- coding: utf-8 -*-
import random
from collections import defaultdict
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, Optional, SupportsFloat, Tuple

import gym
import onnxruntime as ort
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from numpy import asarray, float32

from hackable_engine.board.hex.hex_board import HexBoard, Move
from hackable_engine.common.constants import Game, INF, Print
from hackable_engine.controller import Controller

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
        "d2",
        "d6",
        # "a1c5f5e3d4",
        # "a1d4c5c4e4e3g2",
        # "a1e3d4d3b4",
        # "a1c3e3e4d4",
        # "a1c3d4e2f2",
        # "g7e5d4c6b6",
        # "g7e5c5c4d4",
        # "g7c5d4d5f4",
        # "g7d4e3e4c4c5a6",
        # "g7c5b3e3d4",
        # # same openings twice for counterbalance against single move opening
        # "a1c5f5e3d4",
        # "a1d4c5c4e4e3g2",
        # "a1e3d4d3b4",
        # "a1c3e3e4d4",
        # "a1c3d4e2f2",
        # "g7e5d4c6b6",
        # "g7e5c5c4d4",
        # "g7c5d4d5f4",
        # "g7d4e3e4c4c5a6",
        # "g7c5b3e3d4",
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
    observation_space = gym.spaces.Box(-1, 1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=float32)  # should be int8
    action_space = gym.spaces.Box(ZERO, ONE, shape=(1,), dtype=float32)

    winner: Optional[bool]

    def __init__(self, *args, controller: Optional[Controller] = None, color: bool = True):
        super().__init__()
        self.color = color

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
        self.opening = None

        self.games = 0
        self.scores = defaultdict(lambda: 0)

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        self.best_move: Optional[Tuple[Move, Tuple[float32]]] = None
        self.current_move: Optional[Move] = None

        ext = "Black" if self.color else "White"
        self.opp_model = ort.InferenceSession(
            f"Hex7Cnn{ext}.onnx", providers=["CPUExecutionProvider"]
        )

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(openings), size=self.BOARD_SIZE)
        # don't boot up if there's no need for engine to run, but to train on multiple processes
        # self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        notation = self.controller.board.get_notation()
        print("render: ", self.winner, notation, self._get_reward(self.winner))
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
        self.opening = notation
        self.controller.reset(notation, size=self.BOARD_SIZE, init_move_stack=True)

        if self.controller.board.turn != self.color:
            self._make_self_trained_move(not self.color)
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

                self._make_self_trained_move(not self.color)
                # self._make_random_move()

                winner = self._get_winner()

                # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
                if winner is None:
                    self._prepare_child_moves()

        else:
            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated in the next `step`
            self.controller.board.push(self.current_move)

            winner = self._get_winner()

        self.obs = self.observation_from_board()
        self.winner = winner
        self.steps_done += 1

        reward = self._get_reward(winner, action[0])

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
            return None

        self.games += 1
        self.scores[self.opening] += (1 if winner else -1)

        return winner

    def _get_reward(self, winner, score=None):
        if winner is False:
            # - 1 + ((len(self.controller.board.move_stack) + 1) - 12) / (25 - 12)
            reward = -1 + max(0, (len(self.controller.board.move_stack) - 12)) / 74

        elif winner is True:
            reward = 1 - max(0, (len(self.controller.board.move_stack) - 12)) / 74

        else:
            # return ZERO if score != ZERO and score != ONE else LESS_THAN_ZERO
            return 0  # if score != ZERO and score != ONE else -0.003

        if self.color is False:
            reward = -reward
        return reward

    def _make_random_move(self):
        moves = list(self.controller.board.legal_moves)
        self.controller.board.push(choice(moves))

    def _make_self_trained_move(self, color: bool) -> None:
        best_move: Optional[Move] = None
        best_score = None
        for move in self.controller.board.legal_moves:
            self.controller.board.push(move)
            obs = self.controller.board.as_matrix().astype(float32).reshape(1, 7, 7)
            score = self.opp_model.run(None, {"input": obs})[0][0][0]
            if best_move is None or (color and score > best_score) or (not color and score < best_score):
                if not best_move or random.choice((True, False)):  # randomize a bit which move is selected
                    best_move = move
                    best_score = score
            self.controller.board.pop()

        self.controller.board.push(best_move)

    def observation_from_board(self) -> th.Tensor:
        return th.from_numpy(self.controller.board.as_matrix())

    @staticmethod
    def prepare_action(action):
        return asarray((0, 0, 0, 0, 0, 10, action[0], 1))

    def summarize(self):
        # print("loss summary: ", self.losses)
        print(f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}")
