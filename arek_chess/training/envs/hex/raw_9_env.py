# -*- coding: utf-8 -*-
import os.path
import random
from collections import defaultdict
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, Optional, SupportsFloat, Tuple, List

# import gym
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from numpy import float32
from stable_baselines3.common.policies import ActorCriticPolicy

from arek_chess.board.hex.hex_board import HexBoard, Move
from arek_chess.common.constants import Game, Print
from arek_chess.controller import Controller

LESS_THAN_ZERO: float32 = float32(-0.0003)
MORE_THAN_ZERO: float32 = float32(0.01)
ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

# fmt: off
openings = cycle(
    [
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
        "d2", "d8", "e2", "e8",
    ]
)
# fmt: on


class Raw9Env(gym.Env):
    """"""

    BOARD_SIZE: int = 9
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.Box(
        -1, 1, shape=(BOARD_SIZE, BOARD_SIZE), dtype=float32
    )  # should be int8
    action_space = gym.spaces.Box(ZERO, ONE, shape=(1,), dtype=float32)

    winner: Optional[bool]
    obs: th.Tensor

    policy: Optional[ActorCriticPolicy] = None

    def __init__(
        self, *args, controller: Optional[Controller] = None, color: bool = True, models = []
    ):
        super().__init__()
        self.color = color

        if controller:
            self._setup_controller(controller)
        else:
            self._setup_controller(
                Controller(
                    printing=Print.NOTHING,
                    # tree_params="4,4,",
                    search_limit=9,
                    is_training_run=True,
                    in_thread=False,
                    timeout=3,
                    game=Game.HEX,
                    board_size=self.BOARD_SIZE,
                )
            )
        # self.obs = self.observation_from_board(self.controller.board)

        self.moves_done = 0
        self.winner = None
        self.opening = None

        self.games = 0
        self.scores = defaultdict(lambda: 0)

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        self.moves_list: List[Move] = []
        self.best_move: Optional[Tuple[Move, Tuple[float32]]] = None
        self.current_move: Optional[Move] = None

        self.models = models

    def _setup_controller(self, controller: Controller):
        self.controller = controller
        self.controller._setup_board(next(openings), size=self.BOARD_SIZE)
        # don't boot up if theres no need for engine to run, but to train on multiple processes
        # self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        notation = self.controller.board.get_notation()
        print("render: ", self.winner, notation, self._get_reward(self.winner))
        print("steps done: ", self.moves_done)
        return notation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        super().reset(seed=seed)

        self.render()

        self.opp_model = choice(self.models)

        self.games += 1
        if self.opening:
            self.scores[self.opening] += 1 if self.winner else -1

        notation = next(openings)
        self.opening = notation
        self.controller.reset_board(
            notation, size=self.BOARD_SIZE, init_move_stack=True
        )

        if self.controller.board.turn != self.color:
            self._make_random_move(self.controller.board)
            # self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)
        # obs = self.observation_from_board(self.controller.board)

        # must be after the observation, as it adds a move on the board  !!! WRONG !!!
        self._prepare_child_moves()
        # must precisely be after, because the policy should evaluate the first move candidate (if step_iterative)
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _prepare_child_moves(self) -> None:
        """
        Reset generator and play the first option to be evaluated.
        """

        self.moves = self.controller.board.legal_moves  # returns new generator
        # self.current_move = next(self.moves)
        self.moves_list = list(self.moves)
        self.current_move = self.moves_list[random.randint(0, len(self.moves_list) - 1)]
        self.moves_list.remove(self.current_move)

        self.best_move = None
        self.controller.board.push(self.current_move)

    def _step_aggregated(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Perform policy evaluation on each legal move before finishing the step.

        Takes significantly more RAM as each environment independently loads policy weights.

        Action is disregarded because gives a score of an initial position (not of any actual move to be analysed).
        """

        best_move: Tuple[float32, Move, th.Tensor] = (
            action[0],
            self.current_move,
            self.obs,
        )
        self.controller.board.pop()  # popping the current move which is on the board since reset

        for move in self.moves:
            observation, score = self._get_obs_and_score(move)

            if not best_move or score > best_move[0]:
                best_move = (score, move, observation)

        scr, _, obs = best_move
        winner = self.controller.board.winner()

        # if the last move didn't conclude the game, now play opponent move
        if winner is None:
            # self.controller.make_move(DEFAULT_ACTION)
#            self._make_random_move(self.controller.board)
            self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

            winner = self.controller.board.winner()

            # if the opponent move didn't conclude, reset move generator and play the first option to be evaluated
            if winner is None:
                self._prepare_child_moves()

        reward = self._get_reward(winner, scr)

        self.winner = winner
        self.moves_done += 1

        return obs, reward, winner is not None, False, {}

    def _step_legacy(
        self,
        action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
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
            # self.current_move = next(self.moves)

            if not self.moves_list:
                raise StopIteration
            self.current_move = self.moves_list[random.randint(0, len(self.moves_list) - 1)]
            self.moves_list.remove(self.current_move)

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

            winner = self.controller.board.winner()

            # if the last move didn't conclude the game, now play opponent move and reset generator
            if winner is None:
                # playing against a configured action or an action from self-trained model
                # obs = reshape(self.observation_from_board().astype(float32), (1, 49))
                # opp_action, value = self.opp_model.run(None, {"input": obs})
                # self.controller.make_move(self.prepare_action(opp_action[0]))

                # self.controller.make_move(DEFAULT_ACTION, search_limit=choice((0, 8)))

                self._make_random_move(self.controller.board)
                # self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

                winner = self.controller.board.winner()

                # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
                if winner is None:
                    self._prepare_child_moves()

        self.obs = self.observation_from_board(self.controller.board)

        self.winner = winner
        reward = self._get_reward(winner)
        self.moves_done += 1

        return self.obs, reward, winner is not None, False, {}

    def _get_obs_and_score(self, move: Move) -> Tuple[th.Tensor, float32]:
        """"""

        self.controller.board.push(move)
        observation = self.observation_from_board(self.controller.board)
        score = self.policy(observation.reshape(1, self.BOARD_SIZE, self.BOARD_SIZE).to(th.bfloat16))[0][0]
        self.controller.board.pop()
        return observation, score

    def _step_iterative(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        self.update_best_and_current_move(action)

        observation, winner = self.static_step_part(
            (
                self.controller.board,
                self.current_move,
                self.best_move,
                self.opp_model,
                self.color,
            )
        )

        return self.dynamic_step_part(action, observation, winner)

    def update_best_and_current_move(self, action):
        """"""

        if self.best_move is None:
            self.best_move = (self.current_move, action[0])
        else:
            best_score = self.best_move[1]
            current_score = action[0]

            # white searching for the highest scores, black for the lowest scores
            if (self.color and current_score > best_score) or (
                not self.color and current_score <= best_score
            ):
                self.best_move = (self.current_move, action[0])

        try:
            self.current_move = next(self.moves)
        except StopIteration:
            self.current_move = None

    @classmethod
    def static_step_part(cls, arg_tuple):
        """
        Allow delegating heavy-lifting to a parallel process.

        Making board operations and especially checking if there is a winner is the heaviest part of the step.
        """

        board, current_move, best_move, opp_model, color = arg_tuple

        winner = None
        if current_move is not None:
            # undo last move that was just evaluated
            board.pop()

            # push a new move to be evaluated in the next `step`
            board.push(current_move)
        else:
            # all moves evaluated - undo the last and push the best move on the board
            winner = cls._make_best_move_and_opp_move(board, best_move, opp_model, color)

        return cls.observation_from_board(board), winner

    def dynamic_step_part(self, action, observation, winner):
        """"""

        self.winner = winner
        self.moves_done += 1

        # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
        if self.current_move is None and winner is None:
            self._prepare_child_moves()

        reward = self._get_reward(winner, action[0], self.current_move is None)

        # return self.obs, reward, winner is not None, False, {}
        return observation, reward, winner is not None, False, {}

    @classmethod
    def _make_best_move_and_opp_move(cls, board, best_move, opp_model, color):
        """"""

        board.pop()
        board.push(best_move[0])

        winner = board.winner()

        # if the last move didn't conclude the game, now play opponent move
        if winner is None:
            # self.controller.make_move(DEFAULT_ACTION)

            # cls._make_random_move(board)
            cls._make_self_trained_move(board, opp_model, not color)

            winner = board.winner()

        return winner

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """"""

        # return self._step_iterative(action)
        # return self._step_aggregated(action)
        return self._step_legacy(action)

    def _get_reward(self, winner: Optional[bool], score: Optional[float32] = None, bonus_eligible: bool = False):
        n_moves = len(self.controller.board.move_stack)
        if winner is False:
            reward = -1  # + max(0, (n_moves - 18)) / 126

        elif winner is True:
            reward = 1 - max(0, (n_moves - 18)) / 126

        else:
            bonus = ZERO
            if bonus_eligible:
                if n_moves % 6 in (0, 1):
                    c = self.controller.board.get_shortest_missing_distance(self.color)
                    nc = self.controller.board.get_shortest_missing_distance(not self.color)
                    c_closer_by = nc - c
                    bonus = MORE_THAN_ZERO * c_closer_by  # eliminates `nan`, although should never be `nan`
                # elif n_moves <= 20 and n_moves % 4 in (0, 1):
                #     c = self.controller.board.get_shortest_missing_distance_perf(self.color)
                #     nc = self.controller.board.get_shortest_missing_distance_perf(not self.color)
                #     c_closer_by = nc - c
                #     bonus = MORE_THAN_ZERO * c_closer_by if c_closer_by >= 3 else ZERO
            return (ZERO if score != ZERO and score != ONE else LESS_THAN_ZERO) + bonus

        if not self.color:
            reward = -reward

        return reward

    def _get_reward_for_whos_closer(self):
        """"""

        diff = self.controller.board.get_shortest_missing_distance_perf(
            self.color
        ) - self.controller.board.get_shortest_missing_distance_perf(not self.color)

        if diff > 1:
            return MORE_THAN_ZERO
        else:
            return ZERO

    @staticmethod
    def _make_random_move(board):
        moves = list(board.legal_moves)
        board.push(choice(moves))

    @staticmethod
    def _make_self_trained_move(board, opp_model, opp_color: bool) -> None:
        moves = []
        obss = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            obss.append(board.as_matrix().astype(float32))
            board.pop()

        scores = np.asarray(opp_model.run(None, {"inputs": np.stack(obss, axis=0)})).flatten()

        best_move: Optional[Move] = None
        best_score = None
        for move, score in zip(moves, scores):
            if (
                best_move is None
                or (opp_color and score > best_score)
                or (not opp_color and score < best_score)
            ):
                best_move = move
                best_score = score

        board.push(best_move)

    @staticmethod
    def observation_from_board(board) -> th.Tensor:
        return th.from_numpy(
            board.as_matrix()
        )  # .reshape(1, 1, self.BOARD_SIZE, self.BOARD_SIZE))

    def summarize(self):
        print(
            f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}"
        )
