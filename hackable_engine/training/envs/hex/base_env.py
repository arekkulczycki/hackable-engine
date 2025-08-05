from collections import deque
from random import choice, shuffle, choices
from typing import Any, Generator, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from onnxruntime import InferenceSession

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard, Move
from hackable_engine.common.constants import FLOAT_TYPE

# TODO: investigate why multiplying * 100 changed(equalized) proportion between
#  mean_reward and mean_return, as they should have been proportional before too
ZERO = FLOAT_TYPE(0)
ONE = FLOAT_TYPE(1)
TWO = FLOAT_TYPE(2)
MINUS_ONE = FLOAT_TYPE(-1)
MINUS_TWO = FLOAT_TYPE(-2)


class BaseEnv(gym.Env):
    """"""

    ENV_NAME: str = "raw1hex"
    REWARDS: dict[Optional[bool], FLOAT_TYPE] = {
        None: ZERO,
        True: ONE,
        False: MINUS_ONE,
    }

    reward_range = (REWARDS[False], REWARDS[True])
    """
    Maximum and minimum sum aggregated over an entire episode, not just the final reward. 
    """  # TODO: aggregated really?

    observation_space = gym.spaces.Box(-1, 1, shape=(1, 1, 1), dtype=FLOAT_TYPE)  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=FLOAT_TYPE)

    winner: Optional[bool]
    obs: np.ndarray  # th.Tensor
    reward: FLOAT_TYPE

    def __init__(
        self,
        *,
        render_mode=None,
        color: bool = True,
        models: list | None = None,
        process_id: int | None = None,
        env_id: int | None = None,
        num_processes: int | None = None,
        num_envs: int | None = None,
    ):
        """"""

        super().__init__()
        self.BOARD_SIZE: int = 1
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.MIN_MOVES: int = self.BOARD_SIZE * 2
        self.MAX_ADDITIONAL_MOVES: int = self.MAX_MOVES - self.MIN_MOVES
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        self.OPENINGS: list = []

        self.color: bool = color
        """Color of the agent."""
        self.models: list | None = models
        self.process_id: int | None = process_id
        self.env_id: int | None = env_id
        self.num_processes: int | None = num_processes
        self.num_envs: int | None = num_envs

        self.board = HexBoard(size=self.BOARD_SIZE)

        self.winner: bool | None = None
        """Color of the player that won the game."""
        self.opening = None
        self.generations = 0

        self.games: int = 0

        self.last_intermediate_score: FLOAT_TYPE = FLOAT_TYPE(0.0)
        self.auxiliary_reward: FLOAT_TYPE = FLOAT_TYPE(0.0)
        # self.intermediate_rewards = []

        self.moves: Generator[Move, None, None] = HexBoard.generate_nothing()
        # self.moves_list: List[Move] = []
        self.best_move: tuple[Move, FLOAT_TYPE] | None = None
        self.current_move: Move | None = None

        self.did_force_stop: bool = False
        self.results: deque[int] = deque(maxlen=25)
        self.opp_ort_session = self.models and choice(self.models)
        self.opponent_move_random = False

    def render(self, mode="human", close=False) -> RenderFrame:
        """"""

        n = self.board.size_square - self.board.unoccupied.bit_count()
        if n:
            if self.winner is None:
                print("environment reset before game finished")
                return ""

            notation = self.board.get_notation()
            # print(f"player: {self.color}, winner: {self.winner}", " ".join(self.intermediate_rewards), self.reward)
            print(f"player: {self.color}, winner: {self.winner}", notation, self.reward)
            return notation
        return ""

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        logits: np.ndarray | None = None,
        opening: str | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """"""

        super().reset(seed=seed)
        self.render()

        # self.intermediate_rewards.clear()

        self.winner = None
        self.generations = 0
        self.last_intermediate_score = 0.0

        # winner = self.board.winner_no_turn()
        # if winner is not None:
        self.opening = opening or choice(self.OPENINGS)
        self.board = HexBoard(size=self.BOARD_SIZE, notation=self.opening, init_move_stack=True)

        if self.board.turn != self.color:
            self._make_opponent_move(1, logits)

        # must be last, because the policy should evaluate the first move candidate
        self.obs = self.observation_from_board(self.board)
        return self.obs, {
            "action": 0,
            "winner": None,
            "reward": ZERO,
        }

    def _prepare_child_moves(self) -> None:
        """Reset generator and play the first option to be evaluated."""

        # if self.best_move:
        #     print(f"chosen action with value {self.best_move[1]} from among values: {'|'.join((str(value) for value in self.action_values))}")
        self.best_move = None
        # self.action_values.clear()

        shuffled_moves = list(self.board.legal_moves)
        shuffle(shuffled_moves)
        self.moves = (move for move in shuffled_moves)  # returns new generator
        self.current_move = next(self.moves)

        self.board.push(self.current_move)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        n_moves = len(self.board.move_stack)
        if self.winner is not None:
            return self.obs, ONE if self.winner else MINUS_ONE, True, False, {}

        score = action[0]
        self.generations += 1

        if self.best_move is None:
            self.best_move = (self.current_move, score)
        else:
            best_score = self.best_move[1]
            if (self.color and score > best_score) or (not self.color and score < best_score):
                self.best_move = (self.current_move, score)

        try:
            # if the move gets maximum score then stop searching for a better one
            best_score = self.best_move[1]
            force_stop_threshold = 0.0
            if best_score >= force_stop_threshold and self.sigmoid_random(best_score, force_stop_threshold):
                self.did_force_stop = True
                raise StopIteration
            # if doesn't raise then there are still moves to be evaluated
            self.current_move = next(self.moves)
        except StopIteration:
            winner, reward = self._on_stop_iteration(n_moves)
            final_selection = True
            self.did_force_stop = False
        else:
            # undo last move that was just evaluated
            self.board.pop()

            # push a new move to be evaluated for the next `step`
            self.board.push(self.current_move)

            winner = None
            reward = self._get_intersequence_reward(score)
            final_selection = False

        self.obs = self.observation_from_board(self.board)

        if final_selection and n_moves % self.MAX_MOVES <= 1:
            winner = self.color if reward > 0 else not self.color
            reward = min(ONE, max(MINUS_ONE, reward))

        self.winner = winner
        self.reward = reward
        if winner is not None:
            self.results.append(int(winner == self.color))

        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": score,
                "winner": winner == self.color,
                "reward": reward if winner is not None else ZERO,
                # "opening": self.opening,
            },
        )

    def step_from_logits(self, logits) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        move, score = self.get_move_from_logits(logits)
        self.board.push(move)

        winner, reward = self._get_winner_and_reward(len(self.board.move_stack))

        self.winner = winner
        self.reward = reward

        self.obs = self.observation_from_board(self.board)
        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": score,
                "winner": winner == self.color,
                "reward": reward if winner is not None else ZERO,
                # "opening": self.opening,
            },
        )

    def _get_winner_and_reward(self, n_moves, logits: np.ndarray | None = None):
        winner = self.board.winner_no_turn()
        reward = self._get_reward(winner, n_moves)

        # if the last move didn't conclude the game, now play opponent move
        if winner is None:
            self._make_opponent_move(n_moves, logits)
            winner = self.board.winner_no_turn()

        if winner is not None:
            # overwrite the reward because a winner was found
            reward = self._get_reward(winner, n_moves)

        return winner, reward

    def _get_reward(self, winner: Optional[bool], n_moves: int) -> FLOAT_TYPE:
        if winner is False:
            penalty = self._game_length_penalty(n_moves)
            # reward = (MINUS_ONE + penalty) if self.color else (ONE - penalty)

            # use the following if the games are very long filling whole board (high marginal gain for long games)
            # reward = (MINUS_ONE + penalty**2) if self.color else (ONE - penalty**0.5)

            # use the following if the games are very short (high marginal gain for short games)
            reward = (MINUS_TWO + 2 * penalty**0.5) if self.color else (ONE - penalty**2)

        elif winner is True:
            penalty = self._game_length_penalty(n_moves)
            # reward = (ONE - penalty) if self.color else (MINUS_ONE + penalty)

            # use the following if the games are very long filling whole board
            # reward = (ONE - penalty**0.5) if self.color else (MINUS_ONE + penalty**2)

            # use the following if the games are very short
            reward = (ONE - penalty**2) if self.color else (MINUS_TWO + 2 * penalty**0.5)

        else:
            reward = ZERO
            # reward = self._get_intermediate_reward(n_moves) + self.auxiliary_reward
            # self.auxiliary_reward = FLOAT_TYPE(0.0)

        return reward
        # return FLOAT_TYPE(reward)

    def _get_intermediate_reward(self, n_moves):
        return ZERO
        # return np.FLOAT_TYPE(self._get_intermediate_reward_absolute(n_moves))
        # return FLOAT_TYPE(self._get_intermediate_reward_relative(n_moves))

    def _get_intermediate_reward_absolute(self, n_moves):
        score = self._get_distance_score(n_moves, early_finish=False)

        if (self.color and score > 0) or (not self.color and score < 0):
            return True

        return False

    def _get_intermediate_reward_relative_perf(self, n_moves) -> bool:
        score = self._get_distance_score_perf(n_moves, early_finish=False)
        relative_score = score - self.last_intermediate_score
        self.last_intermediate_score = score

        if (self.color and relative_score > 0) or (not self.color and relative_score < 0):
            return True
        return False

    def _get_intermediate_reward_relative(self, n_moves) -> bool:
        score = self._get_distance_score(n_moves, early_finish=False)
        relative_score = score - self.last_intermediate_score
        self.last_intermediate_score = score

        if (self.color and relative_score > 0) or (not self.color and relative_score < 0):
            return True
        return False

    def _get_intersequence_reward(self, score):
        return ZERO

    def _game_length_penalty(self, n_moves: int) -> float:
        """The more moves are played the higher the punishment."""

        # return n_moves / self.MAX_MOVES
        return (max(0, (n_moves - self.MIN_MOVES)) / self.MAX_ADDITIONAL_MOVES)

    def _get_distance_score(self, n_moves: int) -> FLOAT_TYPE:
        """
        Objective score, i.e. positive is white advantage, negative black advantage.
        Scaled to (-1, 1).

        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        (
            white_missing,
            white_variants,
        ) = self.board.get_short_missing_distances_perf_cached(
            True, should_subtract=n_moves % 2 == 1
        )  # subtracts distance from white because has 1 stone less on board, on odd moves
        (
            black_missing,
            black_variants,
        ) = self.board.get_short_missing_distances_perf_cached(False)

        white_score = sum((self._weight_distance(self.BOARD_SIZE - k, n_moves) * v) for k, v in white_variants.items())
        black_score = sum((self._weight_distance(self.BOARD_SIZE - k, n_moves) * v) for k, v in black_variants.items())

        if not white_score and not black_score:
            return ZERO

        return np.tanh((white_score - black_score) / (white_score + black_score) * 10).astype(FLOAT_TYPE)

    def _get_distance_score_perf(self, n_moves: int) -> FLOAT_TYPE:
        """
        Objective score, i.e. positive is white advantage, negative black advantage, valueswithin -1 and 1.

        Only if is closer by a margin larger than 1, to eliminate the first move advantage bonus.
        """

        white_missing = self.board.get_shortest_missing_distance_perf_cached(True)
        black_missing = self.board.get_shortest_missing_distance_perf_cached(False)

        if not white_missing:
            return ONE
        if not black_missing:
            return MINUS_ONE
        return np.tanh((black_missing - white_missing) / 2).astype(FLOAT_TYPE)

    def _weight_distance(self, distance, n_moves) -> int:
        """Calculate weighted value of distance. In the endgame close connections value more."""

        if n_moves > self.MAX_MOVES / 2:
            return distance**3 / self.MAX_MOVES
        elif n_moves > self.MAX_MOVES / 4:
            return distance**2 / self.BOARD_SIZE
        else:
            return distance

    def _make_opponent_move(self, n_moves, logits: np.ndarray | None = None):
        # self._make_logical_move()
        self._make_random_move()
        # win_percentage = (
        #     np.mean(self.results) if len(self.results) >= N_ENVS / 2 else 1  # 0.4
        # )
        # square = (1 - win_percentage) ** 2
        # if choices([True, False], weights=(1 - win_percentage, win_percentage)):
        #     self._make_random_move(self.board)
        # else:
        #     self._make_logical_move(self.board)

        # self._make_logical_move(self.board)
        # self._make_self_trained_move(
        #     self.board, self.opp_model, not self.color
        # )

    def _make_random_move(self) -> Move:
        """"""

        moves = list(self.board.legal_moves)
        move = choice(moves)
        self.board.push(move)
        return move

    def _make_logical_move(self, n_moves: int) -> Move:
        """"""

        opp_color = not self.color
        best_move: Optional[Move] = None
        best_score = None
        for move in self.board.legal_moves:
            if best_move and np.random.choice((True, False)):
                continue  # in order for the opponent to not always play the same move

            self.board.push(move)
            # score = self._get_distance_score(n_moves)
            score = self._get_distance_score_perf(n_moves)
            self.board.pop()
            if best_move is None or (((opp_color and score > best_score) or (not opp_color and score < best_score))):
                best_move = move
                best_score = score

        self.board.push(best_move)
        return best_move

    # def get_logical_move(self) -> tuple[Move, float, int]:
    #     n_moves = len(self.board.move_stack)
    #     color = self.board.turn
    #
    #     best_move: Optional[Move] = None
    #     best_score = None
    #     for move in self.board.legal_moves:
    #         score = self._get_distance_score_perf(n_moves)
    #         # score = self._get_distance_score(n_moves)
    #
    #         if best_move is None or (
    #             ((color and score > best_score) or (not color and score < best_score))
    #         ):
    #             best_move = move
    #             best_score = score
    #     return best_move, best_score, n_moves

    def get_move_from_logits(self, logits):
        best_move: Optional[Move] = None
        best_score = None
        for move in self.board.legal_moves:
            score = logits[move.mask.bit_length() - 1]
            if best_move is None or (score > best_score and choice([True, False])):  # inherent randomization
                best_move = move
                best_score = score

        return best_move, best_score

    def _make_self_trained_move(self, board, opp_model, opp_color: bool) -> None:
        """"""

        moves = []
        obss = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            # TODO: the reshape is for compatibility to graph trained model input, delete this
            obss.append(self.observation_from_board(board).reshape(self.BOARD_SIZE**2, 1))
            # obss.append(self.observation_from_board(board))
            board.pop()

        scores = opp_model([np.stack(obss, axis=0)])[0].flatten()

        best_move: Optional[Move] = None
        best_score = None
        for move, score in zip(moves, scores):
            if best_move is None or (opp_color and score > best_score) or (not opp_color and score < best_score):
                best_move = move
                best_score = score

        board.push(best_move)

    @staticmethod
    def sigmoid_random(score, force_stop_threshold) -> bool:
        return choices(
            [True, False],
            weights=(p := BaseEnv.sigmoid(score, force_stop_threshold), 1 - p),
        )[0]

    @staticmethod
    def sigmoid(x, force_stop_threshold: float | None = None):
        """
        For threshold 0.5:
            0.5->0.0025, 0.6->0.027, 0.7->0.23, 0.75->0.5, 0.8->0.77, 0.9->0.97, 0.95->0.99
        For threshold 0.0 starts early and slowly, smoothly reaching 1 at 1.
        """
        if force_stop_threshold == 0.5:
            return 1 / (1 + np.e ** (-24 * x + 18))
        else:
            return 1 / (1 + np.e ** (-13 * x + 9)) * 1.015

    @staticmethod
    def observation_from_board(board) -> np.ndarray:
        """"""

        return board.as_matrix().astype(FLOAT_TYPE)
