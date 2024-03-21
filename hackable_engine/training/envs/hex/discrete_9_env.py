# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import cycle
from random import choice
from typing import Any, Dict, Generator, List, Optional, SupportsFloat, Tuple, TypeVar, Callable

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import register
from nptyping import NDArray, Shape, Float32
from numpy import float32, int8
from stable_baselines3.common.policies import ActorCriticPolicy

from hackable_engine.board import BitBoard
from hackable_engine.board.hex.hex_board import HexBoard, Move
from hackable_engine.controller import Controller

ZERO: float32 = float32(0)
ONE: float32 = float32(1)
MINUS_ONE: float32 = float32(-1)
THREE: float32 = float32(3)

IMBALANCED_SCORE_PENALTIES = {
    0.01: float32(-0.0001),
    0.05: float32(-0.00005),
    0.1: float32(-0.000025),
}
"""
Give at most penalty equal to float32(0.0003) = HIGHEST_DISTRIBUTION_PENALTY * DISTRIBUTION_PENALTY_FACTOR
A 9x9 game can take ~3250 evaluations, so maximum total penalty could sum up to 1 at worst,
"""

BEING_CLOSER_BONUS: float32 = float32(0.05)
BALANCED_POSITION_BONUS: float32 = float32(0.025 / (3/4 * 9))

# fmt: off
openings = cycle(
    [
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
        "d2", "d8", "e2", "e8",
    ]
)
# fmt: on

K = TypeVar("K")
V = TypeVar("V")


class keydefaultdict(dict[K, V]):
    def __init__(self, default_factory: Callable[[K], V]):
        super().__init__()
        self.default_factory = default_factory

    def __missing__(self, key: K) -> V:
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class Discrete9Env(gym.Env):
    """"""

    BOARD_SIZE: int = 9
    REWARDS: Dict[Optional[bool], float32] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, BOARD_SIZE, BOARD_SIZE), dtype=float32
    )  # should be int8
    action_space = gym.spaces.MultiDiscrete([
        9,  # area: board split into 9 areas
        3,  # score: discard / maybe / winning
    ])
    # action_space = gym.spaces.Tuple([
    #     gym.spaces.Discrete(9),  # area, board split into 9 areas
    #     gym.spaces.Box(-1, 1, shape=(1,), dtype=float32),  # score, 9 possible values where 0 is the worst and 8 is the best
    # ])
    high_score = 2
    """The best possible score."""

    maybe_score = 1
    """Score for moves from which later choose at random."""

    area_ranges: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]] = {
        0: ((0, 3), (0, 3)),
        1: ((3, 6), (0, 3)),
        2: ((6, 9), (0, 3)),
        3: ((0, 3), (3, 6)),
        4: ((3, 6), (3, 6)),
        5: ((6, 9), (3, 6)),
        6: ((0, 3), (6, 9)),
        7: ((3, 6), (6, 9)),
        8: ((6, 9), (6, 9)),
    }

    winner: Optional[bool]
    obs: th.Tensor

    policy: Optional[ActorCriticPolicy] = None

    def __init__(
        self, *args, render_mode = None, controller: Optional[Controller] = None, color: bool = True, models: List = []
    ):
        """"""

        super().__init__()
        self.color = color
        self.models = models if isinstance(models, List) else models()
        self.agent_num = 1
        self.parallel_env_num = 1
        self.env_name = "raw9hex"

        if controller:
            self.controller = controller
        else:
            self.controller = Controller.configure_for_hex(
                board_kwargs=dict(size=self.BOARD_SIZE),
                is_training=True,
            )
        # self.obs = self.observation_from_board(self.controller.board)

        self.moves_done = 0
        self.winner = None
        self.opening = None
        self.generations = 0

        self.games = 0
        self.scores = defaultdict(lambda: 0)

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        self.maybe_moves: List[Move] = []
        self.best_move: Optional[Tuple[Move, int]] = None
        self.current_move: Optional[Move] = None

        # area_generators: keydefaultdict[int, Generator[Move, None, None]] = keydefaultdict(
        #     lambda area: self.controller.board.generate_moves_from_area(*self.area_ranges[area])
        # )

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        notation = self.controller.board.get_notation()
        print("render: ", self.winner, notation)  #, self._get_reward(self.winner, None, True))
        # print("moves done: ", self.moves_done)
        return notation

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """"""

        super().reset(seed=seed)

        notation = next(openings)
        self.opening = notation

        self.render()

        self.winner = None
        self.opp_model_version = choice([i for i in range(len(self.models))])
        self.opp_model = self.models[self.opp_model_version]

        self.controller.reset_board(
            notation=notation, size=self.BOARD_SIZE, init_move_stack=True
        )

        if self.controller.board.turn != self.color:
            # self._make_random_move(self.controller.board)
            self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

        # start with a random move
        move = self.controller.board.get_random_move()
        self.controller.board.push(move)
        self.current_move = move
        self.maybe_moves = []
        self.best_move = None
        self._set_area_generators(move.mask)

        # the policy should evaluate the position at the end of `reset`
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _set_area_generators(self, visited: BitBoard = 0):
        self.area_generators = {
            key: self.controller.board.generate_moves_from_area(col_range, row_range, visited)
            for key, (col_range, row_range) in self.area_ranges.items()
        }

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """"""

        winner = None
        area = action[0]
        score = action[1]

        if score >= self.high_score:  # and choice([True, False]):
            is_high_score = True
        else:
            winner, is_high_score = self._step_for_low_score(area, score)

        if is_high_score:
            # if all low-score moves already investigated then the highest score is played
            winner = self._step_for_high_score()
            self.maybe_moves = []
            self.best_move = None

        reward = self._get_reward(winner, is_high_score)
        if np.isclose(abs(reward), 1):
            winner = self.color if reward > 0 else not self.color

        self.winner = winner

        return self.obs, reward, winner is not None, False, {"action": score, "winner": winner, "reward": reward, "opening": self.opening, "opp_version": self.opp_model_version}

    def _step_for_high_score(self) -> Optional[bool]:
        """
        Play opponent move if no winner, then if still no winner generate random move for next step.

        Resets the area generators for the next steps because new masks are occupied, unless there is a winner.
        """

        winner = self.controller.board.winner()
        if winner is None:
            self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

            winner = self.controller.board.winner()
            if winner is None:
                move = self.controller.board.get_random_move()
                self.current_move = move
                self.controller.board.push(move)
                self._set_area_generators(move.mask)

                winner = self.controller.board.winner()

        return winner

    def _step_for_low_score(self, area: int, score: int) -> Tuple[Optional[bool], bool]:
        """
        Replace the current move on board with next generated move from selected area.

        If current move score is the highest so far, set it as the best move.
        If all moves have been generated then return to fall back in the high score path.
        """

        self.controller.board.pop()

        if score == self.maybe_score:  # TODO: remove the random choice
            self.maybe_moves.append(self.current_move)

        try:
            self.current_move = next(self.area_generators[area])
        except StopIteration:
            # the net should learn to always put something into `maybe_moves`, however it is not guaranteed
            self.best_move = choice(self.maybe_moves) if self.maybe_moves else self.current_move
            self.controller.board.push(self.best_move)
            is_high_score = True
        else:
            self.controller.board.push(self.current_move)
            is_high_score = False

        winner = self.controller.board.winner()

        return winner, is_high_score

    def _get_reward(self, winner: Optional[bool], is_final_selection: bool = False) -> float32:
        """"""

        n_moves = len(self.controller.board.move_stack)

        if winner is False:
            reward = -1 + self._quick_win_value(n_moves) if self.color else 1 - self._quick_win_value(n_moves)

        elif winner is True:
            reward = 1 - self._quick_win_value(n_moves) if self.color else -1 + self._quick_win_value(n_moves)

        else:
            # the reward when game is being continued, 0 except when we want to punish or encourage some choices
            bonus = ZERO
            if is_final_selection and n_moves > 0:  # and n_moves % 6 == (0 if self.color else 1):  # `n_moves` is `n_moves/2` full moves
                # bonus = self._get_score_for_whos_closer(n_moves, 20)
                bonus = self._get_reward_for_whos_closer(n_moves, 24)
                # bonus = self._principle_bonus()

            # return ZERO
            return bonus
            # return self._imbalanced_score_penalty(score) + bonus if score is not None else bonus

        return float32(reward)

    @staticmethod
    def _quick_win_value(n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        return max(0, (n_moves - 18)) / 126

    def _principle_bonus(self) -> float32:
        """Balanced distribution of stones is always preferred. 3/4 of board size is the highest value."""

        imbalance, central_imbalance = self.controller.board.get_imbalance(self.color)
        return float32(self.controller.board.size * 3 / 4 - imbalance - central_imbalance) * BALANCED_POSITION_BONUS

    @staticmethod
    def _imbalanced_score_penalty(score: float32) -> float32:
        """Negative value of penalty given for a score that is an outsider from a desired score distribution."""

        pessimism = abs(1 - score)
        """How far from the highest possible score the prediction is. 
        The idea is for the model to be pessimistic and evaluate highly rarely, 
        as there are few strong moves, but plenty of losing moves."""

        for threshold, penalty in IMBALANCED_SCORE_PENALTIES.items():
            if pessimism < threshold:
                return penalty
        return ZERO

    def _get_reward_for_whos_closer(self, n_moves: int, move_threshold: int) -> float32:
        """Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus."""

        if n_moves <= move_threshold:
            return ZERO

        color_missing = self.controller.board.get_shortest_missing_distance(self.color)
        opponent_missing = self.controller.board.get_shortest_missing_distance(not self.color)
        c_closer_by = float32(opponent_missing - color_missing)

        # finish game is advantage is sufficient
        if c_closer_by >= 4 and color_missing < c_closer_by:
            return ONE
        if c_closer_by <= -4 and opponent_missing < -c_closer_by:
            return MINUS_ONE

        return ZERO
        # return BEING_CLOSER_BONUS * c_closer_by

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
    def observation_from_board(board) -> NDArray[Shape["1, 9, 9"], Float32]:
        """"""

        return board.as_matrix()

    def summarize(self):
        """"""

        print(
            f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}"
        )


register(
     id="Discrete9Env",
     entry_point="hackable_engine.training.envs.hex.discrete_9_env:Discrete9Env",
)
