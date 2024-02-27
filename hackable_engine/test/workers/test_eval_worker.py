# -*- coding: utf-8 -*-
from itertools import cycle
from multiprocessing import Lock
from time import perf_counter
from unittest import TestCase

from hackable_engine.board.chess.chess_board import ChessBoard
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.training.envs.chess.square_control_env import SquareControlEnv
from hackable_engine.workers.eval_worker import EvalWorker


class TestEvalWorker(TestCase):
    def test_eval_item_speed(self) -> None:
        # fmt: off
        items = cycle(
            [
                EvalItem("id", "name", "move", 0, ChessBoard("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("2b2rk1/1p2q2p/2pbPpp1/p7/2P1B2Q/P7/1B3PPP/4R1K1 w - - 0 22").serialize_position()),
                EvalItem("id", "name", "move", 0, ChessBoard("2b2rk1/1p2q2p/3bPpp1/p2P4/7Q/P7/1B3PPP/4R1K1 b - - 0 23").serialize_position()),
            ]
        )
        # fmt: on

        # speed no model
        worker = EvalWorker(Lock(), Lock(), None, None, 64, 0, ChessBoard)
        worker.setup()

        t0 = perf_counter()
        for i in range(10000):
            ret = worker.eval_item(next(items), None)
        t = perf_counter() - t0
        print(t, f"{10000 / t} per second")

        # fmt: off
        # speed with a model
        worker = EvalWorker(
            Lock(), Lock(), None, None, 64, 0, ChessBoard, model_version="tight-fit.v9", env=SquareControlEnv()
        )
        # fmt: on
        worker.setup()

        t0 = perf_counter()
        for i in range(10000):
            ret = worker.eval_item(next(items), None)
        t = perf_counter() - t0
        print(t, f"{10000 / t} per second")


TestEvalWorker().test_eval_item_speed()
