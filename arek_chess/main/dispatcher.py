# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from arek_chess.board.board import Board
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.messaging import Queue


class Dispatcher:
    """
    Class_docstring
    """

    def __init__(
        self,
        eval_queue: Queue,
        candidates_queue: Queue,
    ):
        self.eval_queue = eval_queue
        self.candidates_queue = candidates_queue

    def dispatch(self, node_name: str, node_turn: bool) -> None:
        board = MemoryManager.get_node_board(node_name)
        board.turn = node_turn

        moves = [move for move in board.legal_moves]

        if moves:
            self.create_node_params_cache(board, node_name)
        else:  # is checkmate
            if board.is_checkmate():
                self.candidates_queue.put(
                    (
                        node_name,
                        [
                            {
                                "node_name": f"{node_name}.0",
                                "move": "checkmate",
                                "i": 0,
                                "captured": 0,
                                "turn": not node_turn,
                                "score": -1000000 if node_turn else 1000000,
                                "is_capture": False,
                            }
                        ],
                    )
                )
            else:
                self.candidates_queue.put(
                    (
                        node_name,
                        [
                            {
                                "node_name": f"{node_name}.0",
                                "move": "stalemate",
                                "i": 0,
                                "captured": 0,
                                "turn": not node_turn,
                                "score": 0,
                                "is_capture": False,
                            }
                        ],
                    )
                )

        moves_n = len(moves)

        put_items = []
        for move in moves:
            captured_piece_type = board.get_captured_piece_type(move)

            put_items.append(
                (
                    node_name,
                    moves_n,
                    move.uci(),
                    not node_turn,
                    captured_piece_type,
                )
            )

        self.eval_queue.put_many(put_items)

    @staticmethod
    def create_node_params_cache(board: Board, node_name) -> None:
        white_params = [
            *board.get_material_and_safety(True),
            *board.get_total_mobility(True),
        ]
        black_params = [
            *board.get_material_and_safety(False),
            *board.get_total_mobility(False),
        ]

        MemoryManager.set_node_params(
            node_name,
            *[white - black for white, black in zip(white_params, black_params)],
        )
