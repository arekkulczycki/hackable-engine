# -*- coding: utf-8 -*-
from collections import defaultdict
from itertools import cycle
from random import choice, shuffle
from typing import Any, Dict, Generator, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.envs.registration import register
from nptyping import NDArray, Shape, Float32
from numpy import float32
from stable_baselines3.common.policies import ActorCriticPolicy

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

BEING_CLOSER_BONUS: float32 = float32(0.005)
BEING_CLOSER_SCORE: float32 = float32(0.01)  # float32(0.005)
BALANCED_POSITION_BONUS: float32 = float32(0.0015 / (3 / 4 * 9))
MAX_MOVES = 81

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

    reward_range = (2 * REWARDS[False], 2 * REWARDS[True])
    """Maximum and minimum sum aggregated over an entire episode, not just the final reward."""

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, BOARD_SIZE, BOARD_SIZE), dtype=float32
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=float32)

    winner: Optional[bool]
    obs: th.Tensor

    policy: Optional[ActorCriticPolicy] = None

    def __init__(
        self,
        *args,
        render_mode=None,
        controller: Optional[Controller] = None,
        color: bool = True,
        models: List = [],
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

        self.moves_done = 0
        self.winner = None
        self.opening = None
        self.generations = 0

        self.games = 0
        self.scores = defaultdict(lambda: 0)

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        # self.moves_list: List[Move] = []
        self.best_move: Optional[Tuple[Move, float32]] = None
        self.current_move: Optional[Move] = None

        self.PRINCIPLE_BONUS_PARAM = self.controller.board.size * 3 / 4

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        n = len(self.controller.board.move_stack)
        if n:
            notation = self.controller.board.get_notation()
            print(
                "render: ", self.winner, notation, self.reward
            )
            return notation
        return ""

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
        self.winner = None
        self.generations = 0

        # winner = self.controller.board.winner_no_turn()
        # if winner is not None:
        notation = next(openings)
        self.opening = notation
        self.controller.reset_board(
            notation=notation, size=self.BOARD_SIZE, init_move_stack=True
        )

        if self.controller.board.turn != self.color:
            # self._make_random_move(self.controller.board)
            self._make_self_trained_move(
                self.controller.board, self.opp_model, not self.color
            )

        self._prepare_child_moves()

        # must be last, because the policy should evaluate the first move candidate
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _prepare_child_moves(self) -> None:
        """Reset generator and play the first option to be evaluated."""

        shuffled_moves = list(self.controller.board.legal_moves)
        shuffle(shuffled_moves)
        self.moves = (move for move in shuffled_moves)  # returns new generator
        self.current_move = next(self.moves)

        self.controller.board.push(self.current_move)

    def _on_stop_iteration(self, score: float32) -> Tuple[Optional[bool], float32]:
        """"""

        # all moves evaluated - undo the last and push the best move on the board
        self.controller.board.pop()
        self.controller.board.push(self.best_move[0])

        winner = self.controller.board.winner_no_turn()
        reward = self._get_reward(winner, True)

        # if the last move didn't conclude the game, now play opponent move and reset generator
        if winner is None:
            # self._make_random_move(self.controller.board)
            self._make_self_trained_move(
                self.controller.board, self.opp_model, not self.color
            )

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
                    # if the opponent move concluded, undo the generated move
                    self.controller.board.pop()
                    assert self.controller.board.turn is self.color

        if winner is not None:
            # overwrite the reward because a winner was found
            reward = self._get_reward(winner, True)

        return winner, reward

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
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        if self.winner is not None:
            return self.obs, ONE if self.winner else MINUS_ONE, True, False, {}

        score = action[0]
        self.generations += 1

        if self.best_move is None:
            self.best_move = (self.current_move, score)
        else:
            best_score = self.best_move[1]
            current_score = score
            # p = 1 if self.generations < 6 else 0.75 if self.generations < 12 else 0.5 if self.generations < 20 else 0.25
            # if current_score > best_score and np.random.choice([True, False], p=[p, 1 - p]):
            if current_score > best_score:
                self.best_move = (self.current_move, score)

        try:
            # if the move gets maximum score then stop searching for a better one
            best_score = self.best_move[1]
            if best_score >= 1.0 or self.generations > 25 and best_score >= 0.75:
                raise StopIteration
            # if doesn't raise then there are still moves to be evaluated
            self.current_move = next(self.moves)
        except StopIteration:
            winner, reward = self._on_stop_iteration(score)
            final_selection = True
        else:
            # continue with the `self.current_move`

            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated in the next `step`
            self.controller.board.push(self.current_move)

            winner = None
            reward = self._get_reward(winner, False)
            final_selection = False

        self.obs = self.observation_from_board(self.controller.board)
        self.winner = winner
        self.reward = reward
        self.best_move = None
        self.moves_done += 1

        if final_selection and len(self.controller.board.move_stack) % MAX_MOVES <= 1:
            winner = self.color if reward > 0 else not self.color
            reward = min(ONE, max(MINUS_ONE, reward))
        # elif final_selection:
        #     print(f"reward: {reward}")

        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": score,
                "winner": winner,
                "reward": reward,
                "opening": self.opening,
            },
        )

    def _get_reward(
        self,
        winner: Optional[bool],
        is_final_selection: bool = False,
    ) -> float32:
        """
        :param bool is_final_selection: if the reward is for the move played on board or for the evaluation cycle
        """

        n_moves = len(self.controller.board.move_stack)

        if winner is False:
            reward = (
                -1 + self._quick_win_value(n_moves)
                if self.color
                else 1 - self._quick_win_value(n_moves)
            )

        elif winner is True:
            reward = (
                1 - self._quick_win_value(n_moves)
                if self.color
                else -1 + self._quick_win_value(n_moves)
            )

        else:
            reward = self._get_intermediate_reward(is_final_selection, n_moves)

        return float32(reward)

    def _get_intermediate_reward(self, is_final_selection, n_moves):
        # the reward when game is being continued, 0 except when we want to punish or encourage some choices
        if (
            is_final_selection and n_moves > 0
        ):  # and n_moves % 6 == (0 if self.color else 1):  # `n_moves` is `n_moves/2` full moves
            freq = 2
            # if MAX_MOVES - (n_moves % MAX_MOVES) <= 1:
            #     return self._get_score_for_whos_closer(
            #         n_moves, early_finish=False
            #     )
            if n_moves > 16:  # % freq in (0, 1):
                reward = self._get_score_for_whos_closer(
                    n_moves, early_finish=False
                ) / MAX_MOVES  # * freq
            else:
                # return ZERO
                reward = self._principle_bonus()

            # val = self.generations / (81 - n_moves)
            # penalty = (BALANCED_POSITION_BONUS if val > 0.99 or val < 0.33 else ZERO)
            return reward  # - penalty

        return ZERO

    def _quick_win_value(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        # return 0.0
        return max(0, (n_moves - 18)) / 120

    def _principle_bonus(self) -> float32:
        """Balanced distribution of stones is always preferred. 3/4 of board size is the highest value."""

        imbalance, central_imbalance = self.controller.board.get_imbalance(self.color)
        return (
            float32(self.PRINCIPLE_BONUS_PARAM - imbalance - central_imbalance)
            * BALANCED_POSITION_BONUS
        )

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
            color_missing = 0  # self.controller.board.get_shortest_missing_distance_perf(self.color)
            opponent_missing = 0  # self.controller.board.get_shortest_missing_distance_perf(not self.color)
        else:
            color_missing = self.controller.board.get_shortest_missing_distance(
                self.color
            )
            opponent_missing = self.controller.board.get_shortest_missing_distance(
                not self.color
            )
        c_closer_by = float32(opponent_missing - color_missing)

        # finish game if advantage is sufficient
        if n_moves > move_threshold:
            if c_closer_by >= 4 and color_missing < c_closer_by:
                return ONE
            if c_closer_by <= -4 and opponent_missing < -c_closer_by:
                return MINUS_ONE
        return ZERO

    def _get_score_for_whos_closer(
        self, n_moves: int, *, early_finish: bool = True
    ) -> float32:
        """
        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        (
            color_missing,
            color_variants,
        ) = self.controller.board.get_short_missing_distances(self.color)
        (
            opponent_missing,
            opponent_variants,
        ) = self.controller.board.get_short_missing_distances(
            not self.color, should_subtract=(not self.color and n_moves % 2 == 1)
        )  # subtracts distance from white because has 1 stone less on board

        # finish game is advantage is sufficient
        if early_finish:
            c_closer_by = float32(opponent_missing - color_missing)
            if c_closer_by >= 4 and color_missing < c_closer_by:
                return ONE
            if c_closer_by <= -4 and opponent_missing < -c_closer_by:
                return MINUS_ONE

        color_score = sum(
            (self._weight_distance(9 - k, n_moves) * v)
            for k, v in color_variants.items()
        )
        opponent_score = sum(
            (self._weight_distance(9 - k, n_moves) * v)
            for k, v in opponent_variants.items()
        )

        return (color_score - opponent_score) * BEING_CLOSER_SCORE

    @staticmethod
    def _weight_distance(distance, n_moves):
        """Calculate weighted value of distance. In the endgame close connections value more."""

        if n_moves > 32:
            return distance**3 / 81
        elif n_moves > 16:
            return distance**2 / 9
        else:
            return distance

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

        scores = np.asarray(
            opp_model.run(None, {"inputs": np.stack(obss, axis=0)})
        ).flatten()

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
    id="Raw9Env",
    entry_point="hackable_engine.training.envs.hex.raw_9_env:Raw9Env",
)
