# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from random import random
from unittest.mock import patch

from numpy import double

from arek_chess.board.board import Board
from arek_chess.common.constants import Print
from arek_chess.controller import Controller
from arek_chess.workers.eval_worker import EvalWorker

evaluated = 0


class MockEvalWorker(EvalWorker):
    def evaluate(
        self, board: Board, move_str: str, captured_piece_type: int, is_check: bool
    ) -> double:
        """"""

        moves = list(move.uci() for move in board.move_stack) + [move_str]
        nm = len(moves)
        sign = -1 if nm % 2 == 0 else 1

        if moves[0] == "e2e4":
            if nm == 2:
                if moves[1] == "e7e6":
                    return double(0.1)
                return double(0.15)
            if nm > 2 and moves[2] == "b1c3":
                return double(0.9 * (0.97) ** nm * sign)
            if nm > 2:
                if "c1g5" in moves:
                    return double(1.1)

                return double(0.8 * (0.99) ** nm * sign)

            return double(1)

        return double(0.0)


class RunWithMockEval:
    """
    Class_docstring
    """

    def prep(self):
        controller = Controller(Print.TREE, "1,3,", 16)
        with patch("arek_chess.controller.EvalWorker", MockEvalWorker):
            controller.boot_up()

        controller.make_move()
        controller.stop_child_processes()


RunWithMockEval().prep()
