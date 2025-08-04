# -*- coding: utf-8 -*-
import math
from random import choices, sample, choice
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv
from hackable_engine.training.envs.util import RealTimeMeanVariance

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)
MINUS_ONEHALF: FLOAT_TYPE = FLOAT_TYPE(-1.5)
MINUS_TWO: FLOAT_TYPE = FLOAT_TYPE(-2)
ZERO_BYTES = (0).to_bytes(22)


class Logit11GraphEnv(BaseEnv):
    """"""

    ENV_NAME = "logit11ghex"

    observation_space = gym.spaces.Box(
        0,
        1,
        shape=(121, 9),
        dtype=FLOAT_TYPE,
        # 0, 1, shape=(3, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8
    # action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)
    action_space = gym.spaces.Discrete(121)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 11
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.MIN_MOVES: int = self.BOARD_SIZE * 2
        self.MAX_ADDITIONAL_MOVES: int = self.MAX_MOVES - self.MIN_MOVES
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        # fmt: off
        self.OPENINGS = [
            "a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11",
            "k1","k2","k3","k4","k5","k6","k7","k8","k9","k10","k11",
            "c2","c10","d2","d10","e2","e10","f2","f10","g2","g10","h2","h10",
            "e3","e9","f3","f9","g3","g9"
        ]
        # fmt: on

        self.rtm = RealTimeMeanVariance()
        # self.opp_ort_session = InferenceSession(choice(self.models), providers=["OpenVINOExecutionProvider"])
        self.opp_ort_session = choice(self.models) if self.models else None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        logits: np.ndarray | None = None,
        opening: str | None = None,
    ) -> tuple[np.array, dict[str, Any]]:
        obs, info = super().reset(seed=seed, logits=logits, opening=opening)

        info["ocb"] = self.board.occupied_co[False].to_bytes(22)
        info["ocw"] = self.board.occupied_co[True].to_bytes(22)
        return obs, info

    @staticmethod
    def observation_from_board(board) -> np.ndarray:
        return board.get_hetero_graph_node_features_one_hot()

    def step(self, move_position):
        """In `BatchedInferenceVectorEnv` the step is not called, but each sub-step separately."""
        n_moves, step_result = self.prepare_step(move_position)
        if step_result:
            self.results.append(0)
            return step_result

        logits = (
            self.opp_ort_session
            and self.opp_ort_session.run(
                None, {"inputs": np.expand_dims(self.observation_from_board(self.board), axis=0)}
            )[0]
        )

        return self.finalize_step(n_moves=n_moves, logits=logits)

    def prepare_step(self, move_position):
        n_moves = self.MAX_MOVES - self.board.unoccupied.bit_count()

        step_result = self._make_selected_move(move_position, n_moves)
        if step_result:  # when illegal move
            self.results.append(int(self.winner == self.color))
            return 0, step_result

        return n_moves, None

    def finalize_step(self, *, n_moves, logits):
        self.winner, self.reward = self._get_winner_and_reward(n_moves, logits)
        if self.winner is not None:
            self.results.append(int(self.winner == self.color))
        self.obs = self.observation_from_board(self.board)
        return (
            self.obs,
            self.reward,
            self.winner is not None,
            False,
            {
                "action": 0,
                "winner": self.winner == self.color,
                "reward": self.reward if self.winner is not None else ZERO,
                "legal": True,
                "ocb": (self.board.occupied_co[False].to_bytes(22) if self.winner is None else ZERO_BYTES),
                "ocw": (self.board.occupied_co[True].to_bytes(22) if self.winner is None else ZERO_BYTES),
                "om": self.opponent_move.c if not self.opponent_move_random else -1,
                # "opening": self.opening,
            },
        )

    def _make_selected_move(self, move_position, n_moves):
        move_pos_int = int(move_position)
        move = Move(mask=1 << move_pos_int, size=self.BOARD_SIZE)
        try:
            self.board.push(move)
        except ValueError:
            # raise
            # print(f"attempting to push {move_position}", move.get_coord())
            # print(f"illegal move {move.get_coord()} in position {self.board.get_notation()}, turn {self.board.turn}")  # process {self.process_id}")
            self.winner = not self.color
            self.reward = MINUS_TWO + self._game_length_penalty(n_moves)
            # self.reward = FLOAT_TYPE(
            #     MINUS_TWO + n_moves / (self.MAX_MOVES - 2 * self.BOARD_SIZE)
            # )
            # TODO: verify if this idea works, penalty and returning back the same position (maybe stop after X attempts?)
            # self.reward = self.PENALTY_PER_ILLEGAL_MOVE
            return (
                self.obs,
                self.reward,
                True,
                False,
                {
                    "action": move_pos_int,
                    "winner": False,
                    "reward": self.reward,
                    "legal": False,
                    "ocb": ZERO_BYTES,
                    "ocw": ZERO_BYTES,
                    "om": -1,
                    # "opening": self.opening,
                },
            )

    def _make_opponent_move(self, n_moves, logits: np.ndarray | None = None):
        minimum_logical_moves = 0.8
        len_results = len(self.results)
        win_percentage = sum(self.results) / len_results if len_results > 0 else minimum_logical_moves
        random_move_weight = (1 - win_percentage)**2 * (1 - minimum_logical_moves)
        # random_move_weight = 1 - math.sin(win_percentage*math.pi/2) ** 0.1
        if (
            n_moves < self.MAX_ADDITIONAL_MOVES  # when board almost filled then finish with logical moves
            and choices((True, False), weights=(random_move_weight, 1 - random_move_weight))[0]
        ):
            move = self._make_random_move()
            self.opponent_move_random = True
        else:
            if logits is None and self.opp_ort_session is not None:
                logits = self.opp_ort_session.run(
                    None, {"inputs": np.expand_dims(self.observation_from_board(self.board), axis=0)}
                )[0]

            if logits is not None:
                move = self._make_opponent_move_from_logits(logits)
                # opponent_move_random set internally
            else:
                move = self._make_logical_move(n_moves + 1)
                self.opponent_move_random = False
        self.opponent_move = move

    def _make_opponent_move_from_logits(self, logits: np.ndarray):
        """The action chosen is the highest value regardless of color, using `argmax` or similar."""
        # deterministic
        # move_c = np.argmax(logits).item()
        # stochastic
        move_c = self._get_softmax_action(logits.flatten(), 0.05)
        move = Move.from_c(move_c, size=self.BOARD_SIZE)
        try:
            self.board.push(move)
        except ValueError:
            move = self._make_random_move()
            self.opponent_move_random = True
        else:
            self.opponent_move_random = False

        return move

    def _get_softmax_action(self, q_values: np.ndarray, a_temperature: float) -> int:
        scaled_qs = q_values / a_temperature
        max_q = np.max(scaled_qs)  # for numerical stability
        exp_qs = np.exp(scaled_qs - max_q)
        probabilities = exp_qs / np.sum(exp_qs)

        action = np.random.choice(self.MAX_MOVES, p=probabilities)
        return action

    def _make_logical_move(self, n_moves: int, color: bool | None = None) -> Move:
        opp_color = color if color is not None else not self.color
        best_move: Move | None = None
        best_score = None
        # n_moves = len(self.board.move_stack) + 1  # after the new move
        legal_moves = list(self.board.legal_moves)
        # shuffle(legal_moves)
        percentage = n_moves / self.MAX_MOVES
        subset_size = max(2, int((self.MAX_MOVES - n_moves) * percentage))
        # print("subset size", subset_size)

        f = self._get_distance_score if n_moves < 3 * self.BOARD_SIZE else self._get_distance_score_perf
        for move in sample(legal_moves, subset_size) if subset_size < len(legal_moves) else legal_moves:
            self.board.push(move)
            try:
                score = f(n_moves)
            except ValueError as e:
                print(e, self.board.occupied_co[True], self.board.occupied_co[False])
                best_move = move
                break

            abs_score = abs(score)
            self.rtm.update(abs_score)
            # break immediately if the score is sufficiently large, saving computation
            if ((opp_color and score > 0) or (not opp_color and score < 0)) and not self.rtm.is_within_half_std(
                abs_score
            ):
                best_move = move
                break

            self.board.pop()
            if best_move is None or (((opp_color and score > best_score) or (not opp_color and score < best_score))):
                best_move = move
                best_score = score
        else:  # runs if no break occurred
            self.board.push(best_move)
        return best_move


register(
    id="Logit11GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_11_graph_env:Logit11GraphEnv",
)
