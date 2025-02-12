# -*- coding: utf-8 -*-
from collections import defaultdict, deque
from itertools import cycle
from random import choice, shuffle, choices
from typing import Any, Dict, Generator, List, Optional, SupportsFloat, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from nptyping import NDArray

from hackable_engine.board.hex.hex_board import HexBoard, Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.controller import Controller

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)

BALANCED_POSITION_BONUS: FLOAT_TYPE = FLOAT_TYPE(0.015 / (3 / 4 * 9))


class BaseRawEnv(gym.Env):
    """"""

    ENV_NAME: str = "raw1hex"
    REWARDS: Dict[Optional[bool], FLOAT_TYPE] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    """
    Maximum and minimum sum aggregated over an entire episode, not just the final reward. 
    """  # TODO: aggregated really?

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 1, 1), dtype=FLOAT_TYPE
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=FLOAT_TYPE)

    winner: Optional[bool]
    obs: NDArray  # th.Tensor

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
        self.BOARD_SIZE: int = 1
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        self.OPENINGS: list = []

        self.color = color
        self.models = models if isinstance(models, List) else models()
        self.agent_num = 1
        self.parallel_env_num = 1

        if controller:
            self.controller = controller
        else:
            self.controller = Controller.configure_for_hex(
                board_kwargs=dict(size=self.BOARD_SIZE),
                is_training=True,
            )

        self.winner = None
        self.opening = None
        self.generations = 0

        self.games = 0
        self.scores = defaultdict(lambda: 0)
        self.last_intermediate_score = 0.0
        # self.intermediate_rewards = []

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        # self.moves_list: List[Move] = []
        self.best_move: Optional[Tuple[Move, FLOAT_TYPE]] = None
        self.current_move: Optional[Move] = None

        self.results = deque(maxlen=100)

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        n = len(self.controller.board.move_stack)
        if n:
            notation = self.controller.board.get_notation()
            # print(f"player: {self.color}, winner: {self.winner}", " ".join(self.intermediate_rewards), self.reward)
            print(f"player: {self.color}, winner: {self.winner}", notation, self.reward)
            return notation
        return ""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """"""

        super().reset(seed=seed)

        self.render()

        # self.intermediate_rewards.clear()

        self.opp_model = choice(self.models)
        self.winner = None
        self.generations = 0
        self.last_intermediate_score = 0.0

        # winner = self.controller.board.winner_no_turn()
        # if winner is not None:
        notation = choice(self.OPENINGS)
        self.opening = notation
        self.controller.reset_board(
            notation=notation, size=self.BOARD_SIZE, init_move_stack=True
        )

        if self.controller.board.turn != self.color:
            self._make_opponent_move(0)

        self._prepare_child_moves()

        # must be last, because the policy should evaluate the first move candidate
        self.obs = self.observation_from_board(self.controller.board)
        return self.obs, {}

    def _prepare_child_moves(self) -> None:
        """Reset generator and play the first option to be evaluated."""

        # if self.best_move:
        #     print(f"chosen action with value {self.best_move[1]} from among values: {'|'.join((str(value) for value in self.action_values))}")
        self.best_move = None
        # self.action_values.clear()

        shuffled_moves = list(self.controller.board.legal_moves)
        shuffle(shuffled_moves)
        self.moves = (move for move in shuffled_moves)  # returns new generator
        self.current_move = next(self.moves)

        self.controller.board.push(self.current_move)

    def step(
        self, action: ActType
    ) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        n_moves = len(self.controller.board.move_stack)
        if self.winner is not None:
            return self.obs, ONE if self.winner else MINUS_ONE, True, False, {}

        score = action[0]
        self.generations += 1

        if self.best_move is None:
            self.best_move = (self.current_move, score)
        else:
            best_score = self.best_move[1]
            if (self.color and score > best_score) or (
                not self.color and score < best_score
            ):
                self.best_move = (self.current_move, score)

        try:
            # if the move gets maximum score then stop searching for a better one
            best_score = self.best_move[1]
            if (
                self.color
                and (
                    best_score >= 1.0
                    or (
                        self.generations > self.MAX_MOVES / 2
                        and best_score >= 0.75
                        # and self.sigmoid_random(best_score)
                    )
                )
            ) or (
                not self.color
                and (
                    (
                        best_score <= -1.0
                        or (
                            self.generations > self.MAX_MOVES / 2
                            and best_score <= -0.75
                            # and self.sigmoid_random(-best_score)
                        )
                    )
                )
            ):
                raise StopIteration
            # if doesn't raise then there are still moves to be evaluated
            self.current_move = next(self.moves)
        except StopIteration:
            winner, reward = self._on_stop_iteration(n_moves)
            final_selection = True
        else:
            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated for the next `step`
            self.controller.board.push(self.current_move)

            winner = None
            reward = ZERO
            final_selection = False

        self.obs = self.observation_from_board(self.controller.board)

        if final_selection and n_moves % self.MAX_MOVES <= 1:
            winner = reward > 0
            reward = min(ONE, max(MINUS_ONE, reward))
        self.winner = winner
        self.reward = reward
        if winner is not None:
            self.results.append(winner == self.color)

        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": score,
                "winner": winner == self.color,
                "reward": reward,
                # "opening": self.opening,
            },
        )

    def _on_stop_iteration(self, n_moves) -> Tuple[Optional[bool], FLOAT_TYPE]:
        """"""

        # all moves evaluated - undo the last and push the best move on the board
        self.controller.board.pop()
        self.controller.board.push(self.best_move[0])
        # self._make_random_move(self.controller.board)

        winner = self.controller.board.winner_no_turn()
        reward = self._get_reward(winner, n_moves)

        # if the last move didn't conclude the game, now play opponent move and reset generator
        if winner is None:
            self._make_opponent_move(n_moves)

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
            reward = self._get_reward(winner, n_moves)

        return winner, reward

    def _get_reward(self, winner: Optional[bool], n_moves: int) -> FLOAT_TYPE:
        """
        :param bool is_final_selection: if the reward is for the move played on board or for the evaluation cycle
        """

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
            reward = self._get_intermediate_reward(n_moves)

        return FLOAT_TYPE(reward)

    def _get_intermediate_reward(self, n_moves):
        # return np.FLOAT_TYPE(self._get_intermediate_reward_absolute(n_moves))
        return np.FLOAT_TYPE(self._get_intermediate_reward_relative(n_moves))

    def _get_intermediate_reward_absolute(self, n_moves):
        score = self._get_distance_score(n_moves, early_finish=False)
        self.last_intermediate_score = score

        if not self.color:
            score *= MINUS_ONE

        return score

    def _get_intermediate_reward_relative(self, n_moves):
        if n_moves <= self.BOARD_SIZE * 2:
            score = self._get_distance_score_perf(n_moves, early_finish=False)
        else:
            score = self._get_distance_score(n_moves, early_finish=False)
        relative_score = score - self.last_intermediate_score
        self.last_intermediate_score = score

        if not self.color:
            relative_score *= MINUS_ONE

        return relative_score

    def _quick_win_value(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        # return 0.0
        return (max(0, (n_moves - 2 * self.BOARD_SIZE)) / self.MAX_MOVES) ** 2

    def _get_distance_score(
        self, n_moves: int, *, early_finish: bool = True
    ) -> FLOAT_TYPE:
        """
        Objective score, i.e. positive is white advantage, negative black advantage, valueswithin -1 and 1.

        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        (
            white_missing,
            white_variants,
        ) = self.controller.board.get_short_missing_distances(
            True, should_subtract=(not self.color and n_moves % 2 == 1)
        )  # subtracts distance from white because has 1 stone less on board, on odd moves
        (
            black_missing,
            black_variants,
        ) = self.controller.board.get_short_missing_distances(False)

        # finish game is advantage is sufficient
        if early_finish:
            white_closer_by = white_missing - black_missing
            if (
                white_closer_by >= self.DECISIVE_DISTANCE_ADVANTAGE
                and white_missing < white_closer_by
            ):
                return ONE
            if (
                white_closer_by <= -self.DECISIVE_DISTANCE_ADVANTAGE
                and black_missing < -white_closer_by
            ):
                return MINUS_ONE

        white_score = sum(
            (self._weight_distance(self.BOARD_SIZE - k, n_moves) * v)
            for k, v in white_variants.items()
        )
        black_score = sum(
            (self._weight_distance(self.BOARD_SIZE - k, n_moves) * v)
            for k, v in black_variants.items()
        )

        if not black_score:
            return ONE
        return np.tanh((white_score / black_score) - 1).astype(np.FLOAT_TYPE)

    def _get_distance_score_perf(
        self, n_moves: int, *, early_finish: bool = True
    ) -> FLOAT_TYPE:
        """
        Objective score, i.e. positive is white advantage, negative black advantage, valueswithin -1 and 1.

        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        white_missing = self.controller.board.get_shortest_missing_distance_perf(True)
        black_missing = self.controller.board.get_shortest_missing_distance_perf(False)

        # finish game is advantage is sufficient
        if early_finish:
            white_closer_by = white_missing - black_missing
            if (
                white_closer_by >= self.DECISIVE_DISTANCE_ADVANTAGE
                and white_missing < white_closer_by
            ):
                return ONE
            if (
                white_closer_by <= -self.DECISIVE_DISTANCE_ADVANTAGE
                and black_missing < -white_closer_by
            ):
                return MINUS_ONE

        if not white_missing:
            return ONE
        return np.tanh((black_missing / white_missing) - 1)

    def _weight_distance(self, distance, n_moves) -> int:
        """Calculate weighted value of distance. In the endgame close connections value more."""

        if n_moves > self.MAX_MOVES / 2:
            return distance**3 / self.MAX_MOVES
        elif n_moves > self.MAX_MOVES / 4:
            return distance**2 / self.BOARD_SIZE
        else:
            return distance

    def _make_opponent_move(self, n_moves):
        # self._make_random_move(self.controller.board)
        win_percentage = np.mean(self.results) if self.results else 0
        n_random_moves = int(self.MAX_MOVES * (1 - win_percentage))
        if n_moves <= n_random_moves:
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)
        # self._make_logical_move(self.controller.board)
        # self._make_self_trained_move(
        #     self.controller.board, self.opp_model, not self.color
        # )

    @staticmethod
    def _make_random_move(board):
        """"""

        moves = list(board.legal_moves)
        board.push(choice(moves))

    def _make_logical_move(self, board):
        """"""

        opp_color = not self.color
        best_move: Optional[Move] = None
        best_score = None
        n_moves = len(self.controller.board.move_stack)
        for move in board.legal_moves:
            if best_move and np.random.choice((True, False)):
                continue  # in order for the opponent to not always play the same move

            score = self._get_distance_score_perf(n_moves)
            if not opp_color:
                score *= MINUS_ONE

            if best_move is None or (
                (
                    (opp_color and score > best_score)
                    or (not opp_color and score < best_score)
                )
            ):
                best_move = move
                best_score = score

        board.push(best_move)

    def _make_self_trained_move(self, board, opp_model, opp_color: bool) -> None:
        """"""

        moves = []
        obss = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            # TODO: the reshape is for compatibility to graph trained model input, delete this
            obss.append(
                self.observation_from_board(board).reshape(self.BOARD_SIZE**2, 1)
            )
            # obss.append(self.observation_from_board(board))
            board.pop()

        scores = opp_model([np.stack(obss, axis=0)])[0].flatten()

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
    def sigmoid_random(score) -> bool:
        return choices(
            [True, False],
            weights=(p := BaseRawEnv.sigmoid(score), 1 - p),
        )[0]

    @staticmethod
    def sigmoid(x):
        """0.5->0.0025, 0.6->0.027, 0.7->0.23, 0.75->0.5, 0.8->0.77, 0.9->0.97, 0.95->0.99"""
        return 1 / (1 + np.e ** (-24 * x + 18))

    @staticmethod
    def observation_from_board(board) -> NDArray:
        """"""

        return board.as_matrix().astype(FLOAT_TYPE)

    def summarize(self):
        """"""

        print(
            f"games: {self.games}, score summary: {sorted(self.scores.items(), key=lambda x: x[1])}"
        )
