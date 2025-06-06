# -*- coding: utf-8 -*-
from random import choices, sample
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


class Logit13GraphEnv(BaseEnv):
    """"""

    ENV_NAME = "logit13ghex"

    observation_space = gym.spaces.Box(
        0,
        1,
        shape=(169, 9),
        dtype=FLOAT_TYPE,
        # 0, 1, shape=(3, 9, 9), dtype=FLOAT_TYPE
    )  # should be int8
    # action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(81,), dtype=FLOAT_TYPE)
    action_space = gym.spaces.Discrete(169)

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 13
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 4
        # fmt: off
        self.OPENINGS = [
            "a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","a12","a13",
            "m1","m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12","m13",
            "c2","c12","d2","d12","e2","e12","f2","f12","g2","g12","h2","h12","i2","i12","k2","k12",
            "b2","l12","f3","g3","h3","i3","f11","g11","h11","i11",
        ]  # 52 openings
        # fmt: on
        self.PENALTY_PER_ILLEGAL_MOVE = 0.5 * self.REWARDS[False] / (self.MAX_MOVES / 2)
        self.MAX_INTERMEDIATE_REWARD = self.REWARDS[True] / (self.MAX_MOVES / 2)

        self.opponent_move: Move | None = None
        self.opponent_move_random: bool = False
        self.counter: int = 0

        self.highest_intermediate_reward: FLOAT_TYPE = FLOAT_TYPE(0.01)

        self.rtm = RealTimeMeanVariance()

        # restrict played openings such that distance caching hits as often as possible
        if self.process_id >= 0:
            self.OPENINGS = list((self.OPENINGS * 3)[
                self.process_id * self.num_envs : (self.process_id + 1) * self.num_envs
            ])
            assert len(self.OPENINGS) == self.num_envs

    def render(self, mode="human", close=False):
        self.counter += 1
        if self.counter % 15 == 0:
            ci = self.board.get_short_missing_distances_cached.cache_info()
            print(
                self.process_id,
                "cache info",
                ci.hits,
                ci.misses,
                round(ci.hits / ci.misses, 2) if ci.misses else 0,
            )

        return super().render()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.array, dict[str, Any]]:
        """"""

        obs, info = super().reset(seed=seed)
        info["ocb"] = self.board.occupied_co[False].to_bytes(22)
        info["ocw"] = self.board.occupied_co[True].to_bytes(22)
        return obs, info

    def _prepare_child_moves(self) -> None:
        return None

    def observation_from_board(self) -> np.ndarray:
        # for GNN
        return self.board.get_hetero_graph_node_features_one_hot()

        # for CNN
        # return self.board.as_matrix()

    def _make_opponent_move(self, n_moves):
        minimum_logical_moves = 0.2
        win_percentage = (
            np.mean(self.results) if len(self.results) >= 5 else minimum_logical_moves
        )
        random_move_weight = (1 - win_percentage) * (1 - minimum_logical_moves) + 0.01
        if choices((True, False), weights=(random_move_weight, 1 - random_move_weight))[
            0
        ]:
            move = self._make_random_move()
            self.opponent_move_random = True
        else:
            move = self._make_logical_move()
            self.opponent_move_random = False
        self.opponent_move = move

    def _make_logical_move(self, color: bool | None = None) -> Move:
        """"""

        opp_color = color if color is not None else not self.color
        best_move: Move | None = None
        best_score = None
        n_moves = len(self.board.move_stack) + 1  # after the new move
        legal_moves = list(self.board.legal_moves)
        # shuffle(legal_moves)
        percentage = n_moves / self.MAX_MOVES
        subset_size = max(2, int((self.MAX_MOVES - n_moves) * percentage))
        # print("subset size", subset_size)

        """searching shortest missing distance on game over 
        748197682631517630355556302808619806679578304793115 
        91155681904663678520187205653140934408286549444"""

        """searching shortest missing distance on game over 
        748245566684308669151461830467086930517087915618464 
        42909242004064430076460384253916899258740756319"""
        f = self._get_distance_score if n_moves < 3 * self.BOARD_SIZE else self._get_distance_score_perf
        for move in sample(legal_moves, subset_size):
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
            if (
                (opp_color and score > 0) or (not opp_color and score < 0)
            ) and not self.rtm.is_within_half_std(abs_score):
                best_move = move
                break

            self.board.pop()
            if best_move is None or (
                (
                    (opp_color and score > best_score)
                    or (not opp_color and score < best_score)
                )
            ):
                best_move = move
                best_score = score
        else:
            self.board.push(best_move)
        return best_move

    def step(self, action):
        step_return = self.step_from_preselected(action, prepare_buffer=False)
        if self.winner is not None:
            self.results.append(float(self.winner == self.color))
        return step_return

    def step_from_preselected(self, move_position, *, prepare_buffer: bool = False):
        n_moves = self.MAX_MOVES - self.board.unoccupied.bit_count()
        logical_move_rate = n_moves / self.MAX_MOVES

        if prepare_buffer:
            for i in range(2):
                if choices(
                    (True, False), weights=(1 - logical_move_rate, logical_move_rate)
                )[0]:
                    self._make_random_move()
                else:
                    self._make_logical_move(self.board.turn)
                winner = self.board.winner_no_turn()
                reward = self._get_reward(winner, n_moves)
                if winner is not None:
                    break
        else:
            move_pos_int = int(move_position)
            move = Move(mask=1 << move_pos_int, size=self.BOARD_SIZE)
            try:
                self.board.push(move)
            except ValueError:
                # raise
                # print(f"attempting to push {move_position}", move.get_coord())
                self.winner = not self.color
                self.reward = MINUS_TWO
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

            winner, reward = self._get_winner_and_reward(n_moves, with_iterations=False)

        self.winner = winner
        self.reward = reward
        self.obs = self.observation_from_board()
        return (
            self.obs,
            reward,
            winner is not None,
            False,
            {
                "action": 0,
                "winner": winner == self.color,
                "reward": reward if winner is not None else ZERO,
                "legal": True,
                "ocb": (
                    self.board.occupied_co[False].to_bytes(22)
                    if winner is None
                    else ZERO_BYTES
                ),
                "ocw": (
                    self.board.occupied_co[True].to_bytes(22)
                    if winner is None
                    else ZERO_BYTES
                ),
                "om": self.opponent_move.c if not self.opponent_move_random else -1,
                # "opening": self.opening,
            },
        )

    # def _get_intermediate_reward(self, n_moves):
    #     score: FLOAT_TYPE = self._get_distance_score(n_moves, early_finish=False)
    #     if not self.color:
    #         score *= -1
    #     return self._normalize_intermediate_reward(score)
    #
    # def _normalize_intermediate_reward(self, score: FLOAT_TYPE):
    #     self.highest_intermediate_reward = max(self.highest_intermediate_reward, abs(score))
    #     return score / self.highest_intermediate_reward * self.MAX_INTERMEDIATE_REWARD


register(
    id="Logit13GraphEnv",
    entry_point="hackable_engine.training.envs.hex.logit_13_graph_env:Logit13GraphEnv",
)
