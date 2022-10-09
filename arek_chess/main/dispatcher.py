"""
Dispatches to the queue nodes to be calculated by EvalWorkers.
"""

from typing import List

from arek_chess.board.board import Board, Move
from arek_chess.main.game_tree.constants import INF
from arek_chess.main.game_tree.node.node import Node
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager


class Dispatcher:
    """
    Dispatches to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(self, eval_queue: QueueManager):
        self.memory_manager = MemoryManager()
        self.eval_queue = eval_queue

        self.dispatched: int = 0

    def dispatch(self, node: Node) -> None:
        node_name = node.name
        try:
            board = self.memory_manager.get_node_board(node_name)
        except ValueError as e:
            # print("eval", os.getpid(), e)
            board = Board()  # TODO: take board from Controller starting position
            for move in node_name.split(".")[1:]:
                board.push(Move.from_uci(move))

            self.memory_manager.set_node_board(node_name, board)
        board.turn = node.color

        moves = [move for move in board.legal_moves]

        if moves:
            moves_n = len(moves)
            self.dispatched += moves_n

            put_items = []
            for move in moves:
                put_items.append(
                    (
                        node_name,
                        moves_n,
                        move.uci(),
                    )
                )

            self.eval_queue.put_many(put_items)

        else:  # is checkmate or stalemate
            node.deep = True

            if board.is_checkmate():
                node.score = -INF if node.color else INF
            else:  # is stalemate
                node.score = 0

            # try:
            #     self.memory_manager.remove_node_board_memory(node.name)
            # except:
            #     import traceback
            #
            #     traceback.print_exc()

    def dispatch_many(self, nodes: List[Node]):
        put_items = []

        for node in nodes:
            node_name = node.name
            try:
                board = self.memory_manager.get_node_board(node_name)
            except FileNotFoundError:
                print(f"fucked up on: {node_name}")
                continue
            # board.turn = node.color

            if node.score in [0, -INF, INF]:
                if board.is_game_over(claim_draw=True):
                    continue

            moves = [move for move in board.legal_moves]

            if moves:
                # self.create_node_params_cache(
                #     board, node_name
                # )  # TODO: remove after switching to delta only

                moves_n = len(moves)
                self.dispatched += moves_n

                for move in moves:
                    put_items.append(
                        (
                            node_name,
                            moves_n,
                            move.uci(),
                        )
                    )

            else:  # is checkmate or stalemate
                # node.deep = True

                if board.is_checkmate():
                    node.score = -INF if node.color else INF
                else:  # is stalemate
                    node.score = 0

                # try:
                #     self.memory_manager.remove_node_board_memory(node.name)
                # except:
                #     import traceback
                #
                #     traceback.print_exc()

        self.eval_queue.put_many(put_items)

    def create_node_params_cache(self, board: Board, node_name) -> None:
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

        self.memory_manager.set_node_params(
            node_name,
            # *params,
            w1 - b1,
            w2 - b2,
            w3 - b3,
            w4 - b4,
            w5 - b5,
        )
