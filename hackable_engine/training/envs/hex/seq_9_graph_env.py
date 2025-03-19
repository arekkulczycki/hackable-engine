# -*- coding: utf-8 -*-
from random import choices
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
from numpy import float32

from hackable_engine.board.hex.move import Move
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)


class Seq9GraphEnv(Raw9Env):
    """"""

    BOARD_SIZE = 9
    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, BOARD_SIZE**2, 1), dtype=float32
    )  # should be int8

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        obs, _ = super().reset(seed=seed, options=options)
        return obs, {
            "action": 0,
            "winner": None,
            "reward": FLOAT_TYPE(0.0),
        }

    def observation_from_board(self) -> np.array:
        """
        :return: tensor with `board.size_square` rows of a single element
        """

        return self.controller.board.get_homo_graph_node_features().reshape((1, 81, 1))

    def _get_intermediate_reward(self, n_moves):
        if self.did_force_stop:
            # if self._get_intermediate_reward_relative(n_moves):
            if self._get_intermediate_reward_absolute(n_moves):
                return FLOAT_TYPE(self.AUXILIARY_REWARD_PER_MOVE)
            return FLOAT_TYPE(- 2 * self.AUXILIARY_REWARD_PER_MOVE)
        return ZERO

    @staticmethod
    def _make_self_trained_move(board, opp_model, opp_color: bool) -> None:
        """"""

        moves = []
        obss = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            obss.append(board.get_graph_node_features().numpy().astype(float32))
            board.pop()

        # if opp_model.get_modelmeta().custom_metadata_map.get("algorithm") == "gat":
        if int(next(iter(opp_model.output(0).names))) > 300:
            # scores = [opp_model.run(None, {"inputs": np.expand_dims(obs, axis=0)})[0][0][0] for obs in obss]
            scores = [opp_model([np.expand_dims(obs, axis=0)])[0] for obs in obss]
        else:
            # scores = np.asarray(
            #     opp_model.run(None, {"inputs": np.stack(obss, axis=0)})
            # ).flatten()
            scores = opp_model([np.stack(obss, axis=0)])[0].flatten()

        best_move: Optional[Move] = None
        best_score = None
        for move, score in zip(moves, scores):
            if best_move is None or score > best_score:
                best_move = move
                best_score = score

        board.push(best_move)

    def _make_opponent_move(self, n_moves):
        win_percentage = np.mean(self.results) if len(self.results) >= 4 else 0.9  # 0.4
        # square = (1 - win_percentage) ** 2
        # if choices([True, False], weights=((1 - win_percentage) / 4, 0.75 + win_percentage/4)):
        if choices(
            [True, False],
            weights=((1 - win_percentage) * 99 / 100, 0.01 + win_percentage),
        ):
            self._make_random_move(self.controller.board)
        else:
            self._make_logical_move(self.controller.board)

    def render(self, mode="human", close=False):
        # return super().render()
        return ""


gym.register(
    id="Seq9GraphEnv",
    entry_point="hackable_engine.training.envs.hex.seq_9_graph_env:Seq9GraphEnv",
)
