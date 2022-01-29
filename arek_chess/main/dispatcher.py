# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from anytree import Node

from arek_chess.board.board import Board
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.messaging import Queue


class Dispatcher:
    """
    Dispatches to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(
        self,
        eval_queue: Queue,
        candidates_queue: Queue,
    ):
        self.eval_queue = eval_queue
        self.candidates_queue = candidates_queue

    def dispatch(self, node: Node, node_turn: bool) -> None:
        node_name = node.name
        board = MemoryManager.get_node_board(node_name)
        try:
            board.turn = node_turn
        except:
            print(node_name)
            print(board)
            raise

        moves = [move for move in board.legal_moves]

        if moves:
            # self.create_node_params_cache(
            #     board, node_name
            # )  # TODO: remove after switching to delta only

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

        else:  # is checkmate or stalemate
            node.deep = True

            if board.is_checkmate():
                node.score = -1000000 if node_turn else 1000000
            else:  # is stalemate
                node.score = 0

            try:
                MemoryManager.remove_node_board_memory(node.name)
            except:
                import traceback
                traceback.print_exc()

    @staticmethod
    def create_node_params_cache(board: Board, node_name) -> None:
        # white_params = [
        #     *board.get_material_and_safety(True),
        #     *board.get_total_mobility(True),
        # ]
        # black_params = [
        #     *board.get_material_and_safety(False),
        #     *board.get_total_mobility(False),
        # ]
        # params = [white - black for white, black in zip(white_params, black_params)]
        w1, w2, w3 = board.get_material_and_safety(True)
        w4, w5 = board.get_total_mobility(True)
        b1, b2, b3 = board.get_material_and_safety(False)
        b4, b5 = board.get_total_mobility(False)

        MemoryManager.set_node_params(
            node_name,
            # *params,
            w1 - b1,
            w2 - b2,
            w3 - b3,
            w4 - b4,
            w5 - b5,
        )
