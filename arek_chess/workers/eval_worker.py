# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import ctypes
import time
import traceback
from signal import signal, SIGTERM
from typing import Tuple

import numpy
from chess import Move
from pyinstrument import Profiler

from arek_chess.board.board import Board
from arek_chess.common_data_manager import CommonDataManager
from arek_chess.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker

# material, safety, under_attack, mobility, king_mobility
# DEFAULT_ACTION = numpy.array([100.0, 1.0, -2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=numpy.float32)
DEFAULT_ACTION = numpy.array([100.0, 1.0, -2.0, 2.0, -1.0], dtype=numpy.float32)


class EvalWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.001

    def __init__(self, eval_queue: Queue, selector_queue: Queue):
        super().__init__()

        self.eval_queue = eval_queue
        self.selector_queue = selector_queue

    def setup(self):
        self.common_data_manager = CommonDataManager()

        self.go_board = ctypes.CDLL("arek_chess/board/go_board/go_board.so")

        self.go_board.get_mobility_and_fen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self.go_board.get_mobility_and_fen.restype = ctypes.c_char_p

        # self.go_board.get_fen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        # self.go_board.get_fen.restype = ctypes.c_char_p

        self.go_board.gc.argtypes = [ctypes.c_char_p]
        self.go_board.gc.restype = ctypes.c_void_p

    def run(self):
        """

        :return:
        """

        self.setup()

        # profiler = Profiler()
        # profiler.start()
        #
        # def before_exit(*args):
        #     """"""
        #     profiler.stop()
        #     profiler.print(show_all=True)
        #     # self.terminate()
        #     exit(0)
        #
        # signal(SIGTERM, before_exit)

        while True:
            eval_item = self.eval_queue.get()
            if eval_item:
                (
                    node_name,
                    size,
                    move,
                    fen_before,
                    turn_after,
                    captured_piece_type,
                ) = eval_item

                # t0 = time.time()
                score, fen_after = self.get_score(
                    node_name, fen_before, turn_after, move, captured_piece_type
                )
                # print(time.time() - t0)

                self.selector_queue.put(
                    (
                        node_name,
                        size,
                        move,
                        fen_after,
                        turn_after,
                        captured_piece_type,
                        score,
                    )
                )
            else:
                time.sleep(self.SLEEP)

    def get_score(
        self,
        node_name: str,
        fen_before: str,
        turn_after: bool,
        move_str: str,
        captured_piece_type: int,
    ) -> Tuple[float, str]:
        """

        :return:
        """

        # key = f"{fen_after.split(' - ')[0]}/{1 if turn_after else 0}"
        #
        # db_value = self.common_data_manager.get_score(key)
        # if db_value is not None:
        #     return float(db_value)

        board, move, moved_piece_type = self.get_move_data(
            move_str, fen_before, not turn_after
        )

        board.push(move)
        fen = board.fen()
        white_mobility, white_king_mobility = board.get_total_mobility(True)
        black_mobility, black_king_mobility = board.get_total_mobility(False)
        board.pop()

        # fen_pointer = self.go_board.get_fen(fen_before.encode(), move_str.encode())
        # fen = fen_pointer.decode()
        # self.go_board.pyfree(fen_pointer)

        # print(fen, white_mobility, black_mobility, self.go_board.get_mobility_and_fen(fen_before.encode(), move_str.encode()).decode().split(","))
        # fen, white_mobility, black_mobility = self.go_board.get_mobility_and_fen(fen_before.encode(), move_str.encode()).decode().split(",")
        # white_mobility, black_mobility = int(white_mobility), int(black_mobility)
        # white_king_mobility, black_king_mobility = 0, 0

        # print(fen, white_mobility, black_mobility, white_mobility_, black_mobility_)

        params = self.common_data_manager.get_params(node_name)
        params[0] += board.get_material_delta(captured_piece_type)
        safety_w = board.get_safety_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        safety_b = board.get_safety_delta(
            False, move, moved_piece_type, captured_piece_type
        )
        under_w = board.get_under_attack_delta(
            True, move, moved_piece_type, captured_piece_type
        )
        under_b = board.get_under_attack_delta(
            False, move, moved_piece_type, captured_piece_type
        )

        params[1] += safety_w - safety_b
        params[2] += under_w - under_b

        params = [
            *params,
            white_mobility - black_mobility,
            white_king_mobility - black_king_mobility,
        ]

        score = board.get_score_from_params(DEFAULT_ACTION, moved_piece_type, params)

        # print(node_name, move_str, white_mobility, black_mobility, safety_w, safety_b, under_w, under_b)
        # white_safety = self.common_data_manager.get_param(node_name, True, "safety")
        # white_safety += board.get_safety_delta(True, move, moved_piece_type, captured_piece_type)
        # black_safety = self.common_data_manager.get_param(node_name, False, "safety")
        # black_safety += board.get_safety_delta(False, move, moved_piece_type, captured_piece_type)
        # safety = (white_safety, black_safety) if white_safety and black_safety else None

        # board.push(move)
        # score = board.get_score(DEFAULT_ACTION, moved_piece_type)

        # self.common_data_manager.set_score(key, score)

        return score, fen

    def get_move_data(
        self, move_str: str, fen_before: str, turn_before: bool
    ) -> Tuple[Board, Move, int]:
        board = Board(fen_before)
        board.turn = turn_before
        move = Move.from_uci(move_str)
        moved_piece_type = board.get_moving_piece_type(move)

        return board, move, moved_piece_type
