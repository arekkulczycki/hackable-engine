# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from numpy import float32
from stable_baselines3.common.policies import ActorCriticPolicy

from hackable_engine.board.hex.hex_board import HexBoard, Move
from hackable_engine.common.constants import Game, Print
from hackable_engine.controller import Controller

MORE_THAN_ZERO: float32 = float32(0.025)
ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

IMBALANCED_SCORE_PENALTIES = {
    0.49: float(-0.0003),
    0.45: float(-0.0002),
    0.4: float(-0.0001),
}
"""
Give at most penalty equal to float32(0.0003) = HIGHEST_DISTRIBUTION_PENALTY * DISTRIBUTION_PENALTY_FACTOR
A 9x9 game can take ~3250 evaluations, so maximum total penalty could sum up to 1 at worst,
"""

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
        self, *args, controller: Optional[Controller] = None, color: bool = True, models: List = []
    ):
        """"""

        super().__init__()
        self.color = color
        self.models = models

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
        self.generations = 0

        self.games = 0
        self.scores = defaultdict(lambda: 0)

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        # self.moves_list: List[Move] = []
        self.best_move: Optional[Tuple[Move, Tuple[float32]]] = None
        self.current_move: Optional[Move] = None

        self.actions: List[float32] = []

    def _setup_controller(self, controller: Controller):
        """"""

        self.controller = controller
        self.controller._setup_board(next(openings), size=self.BOARD_SIZE)
        # don't boot up if theres no need for engine to run, but to train on multiple processes
        # self.controller.boot_up()

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        notation = self.controller.board.get_notation()
        print("render: ", self.winner, notation, self._get_reward(self.winner))
        print("moves done: ", self.moves_done)
        return notation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """"""

        super().reset(seed=seed)

        self.render()

        self.opp_model = choice(self.models)

        # if self.opening:
        #     self.games += 1
        #     self.scores[self.opening] += 1 if self.winner else -1

        notation = next(openings)
        self.opening = notation
        self.controller.reset_board(
            notation, size=self.BOARD_SIZE, init_move_stack=True
        )

        if self.controller.board.turn != self.color:
            # self._make_random_move(self.controller.board)
            self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)
        # obs = self.observation_from_board(self.controller.board)

        # must be after the observation, as it adds a move on the board  !!! WRONG !!!
        self._prepare_child_moves()
        # must precisely be after, because the policy should evaluate the first move candidate (if step_iterative)
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _prepare_child_moves(self) -> None:
        """Reset generator and play the first option to be evaluated."""

        self.moves = self.controller.board.legal_moves  # returns new generator
        self.current_move = next(self.moves)
        # self.moves_list = list(self.moves)
        # self.current_move = self.moves_list[random.randint(0, len(self.moves_list) - 1)]
        # self.moves_list.remove(self.current_move)

        self.best_move = None
        self.generations = 1
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

        legal_moves_exhausted = False

        score = action[0]
        self.actions.append(score)
        self.generations += 1

        if self.best_move is None:
            self.best_move = (self.current_move, score)
        else:
            best_score = self.best_move[1]
            current_score = score
            p = 1 if self.generations < 3 else 0.75 if self.generations < 6 else 0.5 if self.generations < 10 else 0.25
            if current_score > best_score and np.random.choice([True, False], p=[p, 1 - p]):
                self.best_move = (self.current_move, score)

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
            assert self.controller.board.turn is not self.color
            self.controller.board.pop()
            self.controller.board.push(self.best_move[0])
            assert self.controller.board.turn is not self.color

            winner = self.controller.board.winner()

            # if the last move didn't conclude the game, now play opponent move and reset generator
            if winner is None:
                # playing against a configured action or an action from self-trained model
                # obs = reshape(self.observation_from_board().astype(float32), (1, 49))
                # opp_action, value = self.opp_model.run(None, {"input": obs})
                # self.controller.make_move(self.prepare_action(opp_action[0]))

                # self.controller.make_move(DEFAULT_ACTION, search_limit=choice((0, 8)))

                # self._make_random_move(self.controller.board)
                self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

                # reset generator and play the first option to be evaluated, regardless if the game is concluded
                try:
                    self._prepare_child_moves()
                except StopIteration:
                    # can only happen if the black player fills the last empty cell in the board
                    winner = False
                else:
                    # the winner is checked after the move preparation in order to avoid calculating reward for game over
                    winner = self.controller.board.winner_no_turn()

                    if winner is (not self.color):
                        # if the opponent move concluded, undo the generated first move
                        self.controller.board.pop()
                        assert self.controller.board.turn is self.color

            legal_moves_exhausted = True

        self.obs = self.observation_from_board(self.controller.board)
        self.winner = winner
        self.moves_done += 1

        reward = self._get_reward(winner, score, legal_moves_exhausted)

        return self.obs, reward, winner is not None, False, {"action": score, "winner": winner, "reward": ZERO if winner is None else reward, "opening": self.opening, "pseudo_reward": reward if winner is None else ZERO}

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

        is_final_selection = self.current_move is None
        """The algorithm iterates over all legal moves, then goes back to the one evaluated highest."""

        reward = self._get_reward(winner, action[0], is_final_selection)

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

    def _get_reward(self, winner: Optional[bool], score: Optional[float32] = None, is_final_selection: bool = False) -> float32:
        """"""

        n_moves = len(self.controller.board.move_stack)

        if winner is False:
            reward = -1 + self._quick_win_value(n_moves) if self.color else 1 - self._quick_win_value(n_moves)

        elif winner is True:
            reward = 1 - self._quick_win_value(n_moves) if self.color else -1 + self._quick_win_value(n_moves)

        else:
            # the reward when game is being continued, 0 except when we want to punish or encourage some choices
            bonus = ZERO
            if is_final_selection and n_moves > 0 and n_moves % 8 == (0 if self.color else 1):  # `n_moves` is `n_moves/2` full moves
                bonus = self._get_reward_for_whos_closer(n_moves, 33)

            return self._imbalanced_score_penalty(score) + bonus if score is not None else bonus

        return float32(reward)

    def _quick_win_value(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        return max(0, (n_moves - 18)) / 126

    def _imbalanced_score_penalty(self, score: float32):
        """Negative value of penalty given for a score that is an outsider from a desired normal score distribution."""

        # percentile = DESIRED_SCORE_DISTRIBUTION.cdf(score)
        # distance_to_egde = percentile if percentile < 0.5 else 1 - percentile
        # return (-0.5 / distance_to_egde + 1) * DISTRIBUTION_PENALTY_FACTOR

        distance_to_mean = abs(1 - score)
        for threshold, penalty in IMBALANCED_SCORE_PENALTIES.items():
            if distance_to_mean > threshold:
                return penalty
        return ZERO

    def _get_reward_for_whos_closer(self, n_moves: int, move_threshold: int) -> float32:
        """Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus."""

        if n_moves <= move_threshold:
            c = self.controller.board.get_shortest_missing_distance_perf(self.color)
            nc = self.controller.board.get_shortest_missing_distance_perf(not self.color)
        else:
            c = self.controller.board.get_shortest_missing_distance(self.color)
            nc = self.controller.board.get_shortest_missing_distance(not self.color)
        c_closer_by = nc - c
        return MORE_THAN_ZERO * c_closer_by if abs(c_closer_by) > 1 else ZERO

    @staticmethod
    def _make_random_move(board):
        """"""

        moves = list(board.legal_moves)
        board.push(choice(moves))

    @staticmethod
    def _make_self_trained_move(board, opp_model, opp_color: bool) -> None:
        """"""

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
        """"""

        return th.from_numpy(board.as_matrix())

    def summarize(self):
        """"""

        print(
            f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}"
        )
