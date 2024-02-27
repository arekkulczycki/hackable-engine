# -*- coding: utf-8 -*-
from random import choice

from hackable_engine.board.hex.hex_board import HexBoard, Move
from hackable_engine.board.hex.bitboard_utils import generate_masks


class RandomFillingTest:
    """
    A fun excercise to show how random filling of hex board is a wrong indicator of result.
    """

    def fill_at_random(self) -> None:
        """"""

        white_wins = 0

        for i in range(100):
            # initialize a board fully filled except black bridges from top to bottom, sufficient for easy black win
            board = HexBoard("e2d1d4c1c6a7a6b6a5b5a4b4a3b3a2b2a1g1g2f2g3f3f4e4f5e5e6d6e7d7g4d2g5c4g6c3f6c2f7b1g7",
                             size=7)
            empty_masks = list(generate_masks(board.unoccupied))

            winner = None
            while winner is None:
                mask = choice(empty_masks)
                empty_masks.remove(mask)

                board.push(Move(mask, board.size))
                winner = board.winner()

            white_wins += int(winner)

        print(white_wins)


RandomFillingTest().fill_at_random()
