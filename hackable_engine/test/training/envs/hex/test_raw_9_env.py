from unittest import TestCase

from numpy import float32
from parameterized import parameterized

from hackable_engine.board.hex.hex_board import HexBoard
from operator import lt, gt

from hackable_engine.training.envs.hex.raw_9_env import Raw9Env


class HexBoardTestCase(TestCase):
    env: Raw9Env

    def setUp(self):
        self.env = Raw9Env(color=False)  # black player perspective

    @parameterized.expand(
        [
            # ["a1g7e3e6d5c5i5h6b3g6h2i4d4c6f8d7f9f3f6c7", 9, lt],
            # ["a1g7e3e6d5c5i5h6b3g6h2i4d4c6f8d7f9f3f6c7f5i6", 9, gt],
            ["a9d3c3c4f4f7", 9, gt],
            ["a9d3c3c4f4f7g7g6e6g2", 9, gt],
            ["a4g5a8g4h2f5f8g2d6b8d3a1", 9, gt],
        ]
    )
    def test_get_score_for_whos_closer(self, notation, size, operator) -> None:
        """"""

        board = HexBoard(notation=notation, size=size, init_move_stack=True)
        self.env.controller.board = board

        reward = self.env._get_score_for_whos_closer(len(board.move_stack), 0, False)
        print(reward)

        assert operator(reward, 0)
