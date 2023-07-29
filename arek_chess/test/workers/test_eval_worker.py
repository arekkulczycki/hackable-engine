# -*- coding: utf-8 -*-
from itertools import cycle
from time import perf_counter
from unittest import TestCase

from arek_chess.board.board import Board
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.workers.eval_worker import EvalWorker


class TestEvalWorker(TestCase):

    def test_eval_item_speed(self) -> None:
        items = cycle(
            [
                EvalItem("id", "name", "move", 0, Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("2b2rk1/1p2q2p/2pbPpp1/p7/2P1B2Q/P7/1B3PPP/4R1K1 w - - 0 22").serialize_position()),
                EvalItem("id", "name", "move", 0, Board("2b2rk1/1p2q2p/3bPpp1/p2P4/7Q/P7/1B3PPP/4R1K1 b - - 0 23").serialize_position()),
            ]
        )

        worker = EvalWorker(None, None, 0)
        worker.setup()
        t0 = perf_counter()
        for i in range(10000):
            ret = worker.eval_item(next(items))
        t = perf_counter() - t0
        print(t, f"{10000 / t} per second")


TestEvalWorker().test_eval_item_speed()
