# -*- coding: utf-8 -*-
from random import random
from time import sleep
from typing import Optional, Type
from unittest import TestCase
from unittest.mock import patch

from numpy import float32

from hackable_engine.board import GameBoardBase
from hackable_engine.common.constants import Print
from hackable_engine.controller import Controller
from hackable_engine.criteria.evaluation.base_eval import WeightsType
from hackable_engine.workers.eval_worker import EvalWorker

evaluated = 0


class MockEvalWorker(EvalWorker):
    def evaluate(
        self,
        board: GameBoardBase,
        move_str: str,
        captured_piece_type: int,
        is_check: bool,
    ) -> float32:
        """"""

        moves = list(move.uci() for move in board.move_stack) + [move_str]
        nm = len(moves)
        sign = -1 if nm % 2 == 0 else 1

        if moves[0] == "e2e4":
            if nm == 2:
                if moves[1] == "e7e6":
                    return float32(0.1)
                return float32(0.15)
            if nm > 2 and moves[2] == "b1c3":
                return float32(0.9 * (0.97) ** nm * sign)
            if nm > 2:
                if "c1g5" in moves:
                    return float32(1.1)

                return float32(0.8 * (0.99) ** nm * sign)

            return float32(1)

        return float32(0.0)


def set_constants(target):
    def wrap(level, eval):
        target.LEVEL = level
        target.EVAL = eval
        return target

    return wrap


@set_constants
class EvalWorkerFixedOverLevel(EvalWorker):
    level = 7
    eval = float32(1.5)

    def evaluate(
        self,
        board: GameBoardBase,
        move_str: str,
        captured_piece_type: int,
        is_check: bool,
    ) -> float32:
        """"""

        moves = list(move.uci() for move in board.move_stack) + [move_str]
        nm = len(moves)
        if nm > self.level:
            return self.eval

        if nm % 2 == 1:
            return float32(random() + 2)  # between 2 and 3
        else:
            return float32(random() - 3)  # between -3 and -2


class EvalWorkerSpecificPath(EvalWorker):
    # fmt: off
    PATH = {
       "e2e4": (0.02, -1.9), "e7e5": (-2, 1.5), "d2d4": (1.5, -1.5), "d7d5": (2, 1), "f2f4": (2, 1), "f7f5": (2, 1),
       "c2c4": (2, 1), "c7c5": (2, 1), "b2b4": (2, 1), "b7b5": (2, 1), "g2g4": (2, 1), "g7g5": (2, 1),
       "a2a4": (2, 1), "a7a5": (2, 1), "h2h4": (2, 1), "h7h5": (2, 1)
    }
    # fmt: on
    eval = float32(1.5)

    def evaluate(
        self, board: GameBoardBase, is_check: bool, action: Optional[WeightsType]
    ) -> float32:
        """"""

        # moves = list(move.ui() for move in board.move_stack) + [move_str]
        # nm = len(moves)
        # if all([a == b for a, b in zip(reversed(moves), list(self.PATH.keys())[:nm])]):
        #     # self.RETURNED_ONCE = True
        #     if nm % 2 == 1:
        #         return self.eval
        #     else:
        #         return - self.eval

        # sleep(0.0001)
        return float32(random() * 2 - 1)


class RunWithMockEval(TestCase):
    """
    Class_docstring
    """

    @staticmethod
    def search(mock_worker_class: Type[EvalWorker]):
        controller = Controller(
            printing=Print.LOGS,
            tree_params="1,3,",
            search_limit=9,
            in_thread=False,
            is_training_run=True,
        )
        with patch("hackable_engine.controller.EvalWorker", mock_worker_class):
            controller.boot_up()

        # controller.get_move(reuse_node=False)
        for _ in range(10):
            controller.make_move(reuse_node=False)
            # controller.get_move(reuse_node=False)
        # root_node = controller.search_worker.root

        controller.tear_down()
        # return root_node

    def test_propagation(self):
        eval = float32(1.5)

        # for i in range(3, 4):
        mock_worker = EvalWorkerFixedOverLevel(
            3, eval
        )  # returns specific eval in level i
        root_node = self.search(mock_worker)
        print(root_node.score)
        assert root_node.score == eval


RunWithMockEval.search(EvalWorkerSpecificPath)
