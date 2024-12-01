from typing import Optional

import numpy as np
from numpy import float32
import torch as th
import gymnasium as gym

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.board.hex.move import Move
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env


class Raw9GraphEnv(Raw9Env):
    """"""

    BOARD_SIZE = 9
    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, BOARD_SIZE**2, 1), dtype=float32
    )  # should be int8

    @staticmethod
    def observation_from_board(board: HexBoard) -> th.Tensor:
        return th.reshape(board.get_graph_node_features(), (1, 81, 1))

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
            if (
                best_move is None or score > best_score
            ):
                best_move = move
                best_score = score

        board.push(best_move)


gym.register(
     id="Raw9GraphEnv",
     entry_point="hackable_engine.training.envs.hex.raw_9_graph_env:Raw9GraphEnv",
)
