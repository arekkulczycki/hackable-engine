"""
Dispatches to the queue nodes to be calculated by EvalWorkers.
"""

import sys
from signal import SIGTERM, signal
from time import sleep
from typing import List, Tuple

from pyinstrument import Profiler

from arek_chess.board.board import Move, Board
from arek_chess.constants import INF, SLEEP
from arek_chess.utils.queue_manager import QueueManager
from arek_chess.workers.base_worker import BaseWorker


class DispatcherWorker(BaseWorker):
    """
    Dispatches to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(self, dispatcher_queue: QueueManager, eval_queue: QueueManager):
        """"""

        super().__init__()

        self.dispatcher_queue = dispatcher_queue
        self.eval_queue = eval_queue

        self.dispatched: int = 0

    def setup(self):
        """"""

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.max_items_at_once = 256

    def _run(self):
        """"""

        self.setup()

        max_items_at_once = self.max_items_at_once

        while True:
            items: List[Tuple[str, str, float]] = self.get_items(max_items_at_once)
            self.dispatch(items)

    def get_items(self, max_items_at_once: int) -> List[Tuple[str, str, float]]:
        """"""

        while True:
            items = self.dispatcher_queue.get_many(max_items_at_once)
            if not items:
                sleep(SLEEP)
            else:
                return items

    def dispatch(self, items: List[Tuple[str, str, float]]) -> None:
        """"""

        queue_items = []

        for node_name, move_str, score in items:  # TODO: get_many_boards ?
            board = self.create_node_board(node_name, move_str)

            if score in [0, -INF, INF]:
                if board.is_game_over(claim_draw=True):
                    continue

            new_queue_items = [(node_name, move.uci()) for move in board.legal_moves]

            if new_queue_items:
                queue_items += new_queue_items

            # else:  # is checkmate or stalemate  TODO: how to propagate back to tree? is it needed?
            #     if board.is_checkmate():
            #         score = -INF if board.turn else INF
            #     else:  # is stalemate
            #         score = 0

                # try:
                #     self.memory_manager.remove_node_board_memory(node.name)
                # except:
                #     import traceback
                #
                #     traceback.print_exc()

        if queue_items:
            self.dispatched += len(queue_items)
            self.eval_queue.put_many(queue_items)

    def create_node_board(self, node_name: str, node_move: str) -> Board:
        """"""

        board: Board = self.memory_manager.get_node_board(
            ".".join(node_name.split(".")[:-1] or "0")
        )
        if node_move:
            board.push(Move.from_uci(node_move))

        self.memory_manager.set_node_board(node_name, board)

        return board

    def before_exit(self, *args) -> None:
        """"""

        if getattr(self, "should_profile_code", False):
            self.profiler.stop()
            self.profiler.print(show_all=True)

        sys.exit(0)

    def profile_code(self) -> None:
        """"""

        self.profiler = Profiler()
        self.profiler.start()

        self.should_profile_code = True

    # def create_node_params_cache(self, board: Board, node_name) -> None:  TODO: is this idea useful?
    #     # white_params = [
    #     #     *board.get_material_and_safety(True),
    #     #     *board.get_total_mobility(True),
    #     # ]
    #     # black_params = [
    #     #     *board.get_material_and_safety(False),
    #     #     *board.get_total_mobility(False),
    #     # ]
    #     # params = [white - black for white, black in zip(white_params, black_params)]
    #     w1, w2, w3 = board.get_material_and_safety(True)
    #     w4, w5 = board.get_total_mobility(True)
    #     b1, b2, b3 = board.get_material_and_safety(False)
    #     b4, b5 = board.get_total_mobility(False)
    #
    #     self.memory_manager.set_node_params(
    #         node_name,
    #         # *params,
    #         w1 - b1,
    #         w2 - b2,
    #         w3 - b3,
    #         w4 - b4,
    #         w5 - b5,
    #     )
