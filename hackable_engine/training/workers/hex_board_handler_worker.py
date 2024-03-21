from itertools import cycle
from multiprocessing import Process
from random import choice
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from nptyping import NDArray
from numpy import float32

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.queue.manager import QueueManager

# fmt: off
OPENINGS = cycle(
    [
        "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
        "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
        "d2", "d8", "e2", "e8",
    ]
)
# fmt: on


class HexBoardHandlerWorker(Process):

    def __init__(self, in_queue: QueueManager, out_queue: QueueManager, size: int, color: bool, num_envs: int) -> None:
        """
        :param color: the color of AI player, so if white player trains then here color is black (False)
        """

        super().__init__()

        self.in_queue: QueueManager = in_queue
        self.out_queue: QueueManager = out_queue

        self.size = size
        self.color = color
        self.num_envs = num_envs
        self.boards: Dict[int, HexBoard] = {}
        self.models = self._get_models(color)

    @staticmethod
    def _get_models(color) -> List[ort.InferenceSession]:
        models = []
        color_ext = "Black" if color else "White"
        for model_version in ["A", "B", "C", "D", "E", "F"]:
            path = f"Hex9{color_ext}{model_version}.onnx"
            models.append(
                ort.InferenceSession(
                    path, providers=["CPUExecutionProvider"]
                )
            )
        return models

    def run(self) -> None:
        self._init_boards()

        while True:
            item = self.in_queue.queue.get_blocking(timeout=10)  # intention is that timeout will kill the process
            env_id, best_move_str = item
            if best_move_str is None:
                move_str, obs = self._reset_board(env_id)
                winner = None
            else:
                move_str, obs, winner = self._make_move(env_id, best_move_str)

            self.out_queue.put((move_str, obs, winner))

    def _init_boards(self) -> None:
        for i in range(self.num_envs):
            self._reset_board(i)

    def _reset_board(self, env_id: int) -> Tuple[str, NDArray]:
        notation = next(OPENINGS)
        board = HexBoard(notation, size=self.size)
        if board.turn == self.color:
            self._make_self_trained_move(board)

        move = board.get_random_move()
        board.push(move)

        self.boards[env_id] = board
        return move.get_coord(), self._observation_from_board(board)

    def _make_move(self, env_id: int, best_move_str: str) -> Tuple[str, NDArray, Optional[bool]]:
        board: HexBoard = self.boards[env_id]
        next_move = ""

        try:
            next_move = next(board.legal_moves)
        except StopIteration:
            board.pop()
            board.push_coord(best_move_str)
            self._make_self_trained_move(board)

            winner = board.winner()
            if winner is None:
                next_move = board.get_random_move()
                board.push(board.get_random_move())
                winner = board.winner()
        else:
            board.pop()
            board.push(next_move)
            winner = board.winner()

        return next_move.get_coord(), self._observation_from_board(board), winner

    @staticmethod
    def _observation_from_board(board) -> NDArray:
        """"""

        return board.as_matrix()

    def _make_self_trained_move(self, board) -> None:
        """"""

        moves = []
        obss = []
        for move in board.legal_moves:
            moves.append(move)
            board.push(move)
            obss.append(board.as_matrix().astype(float32))
            board.pop()

        scores = np.asarray(choice(self.models).run(None, {"inputs": np.stack(obss, axis=0)})).flatten()

        best_move = None
        best_score = None
        for move, score in zip(moves, scores):
            if (
                best_move is None
                or (self.color and score > best_score)
                or (not self.color and score < best_score)
            ):
                best_move = move
                best_score = score

        board.push(best_move)
