# -*- coding: utf-8 -*-
"""
Wrapper to interface between lichess bot and the engine.
"""

import subprocess as sp

from chess import Move
from chess.engine import PlayResult

from lichess_bot.strategies import MinimalEngine


class HackableBot(MinimalEngine):
    """
    Wrapper to interface between lichess bot and the engine.
    """

    def __init__(self, commands, options, stderr, draw_or_resign, name=None, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, name, **popen_args)

    def search(self, board, time_limit, ponder, draw_offered, root_moves) -> PlayResult:
        """
        The method to be implemented in your homemade engine

        NOTE: This method must return an instance of "chess.engine.PlayResult"
        """

        move = sp.check_output(["./engines/hackable_bot.sh", board.fen(), "16"]).decode().split("\n")[0]

        return PlayResult(move=Move.from_uci(move), ponder=None)
