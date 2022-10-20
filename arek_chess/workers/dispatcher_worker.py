"""
Dispatches to the queue nodes to be calculated by EvalWorkers.
"""

import sys
from signal import SIGTERM, signal
from time import sleep
from typing import List, Tuple, Optional

from pyinstrument import Profiler

from arek_chess.board.board import Move, Board
from arek_chess.common.constants import INF, SLEEP, FINISHED
from arek_chess.common.queue_manager import QueueManager as QM
from arek_chess.workers.base_worker import BaseWorker


class DispatcherWorker(BaseWorker):
    """
    Dispatches to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(self, dispatcher_queue: QM, eval_queue: QM, selector_queue: QM, control_queue: QM, throttle: int):
        """"""

        super().__init__()

        self.dispatcher_queue = dispatcher_queue
        self.eval_queue = eval_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = throttle

        self.dispatched = 0

    def setup(self):
        """"""

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

    def _run(self):
        """"""

        self.setup()

        queue_throttle = self.queue_throttle
        get_items = self.get_items
        dispatch = self.dispatch
        while True:
            items: List[Tuple[str, str, float]] = get_items(queue_throttle)
            if items:
                dispatch(items)
            else:
                sleep(SLEEP)

    def get_items(self, queue_throttle: int) -> List[Tuple[str, str, float]]:
        """"""

        return self.dispatcher_queue.get_many(queue_throttle)

    def dispatch(self, items: List[Tuple[str, str, float]]) -> None:
        """"""

        queue_items = []

        for node_name, move_str, score in items:  # TODO: get_many_boards ?
            if node_name == FINISHED:
                self.dispatched = 0
                return

            try:
                board = self.create_node_board(node_name, move_str)
            except Exception as e:
                print(f"dispatcher error: {e}")
                continue

            if board.is_game_over(claim_draw=False):
                if score != 0:
                    self.selector_queue.put((
                        ".".join(node_name.split(".")[:-1]),
                        move_str,
                        0,
                        -1,  # sending -1 as signal that checkmate is on the board
                        -INF if board.turn else INF,
                    ))
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
            self.control_queue.put(self.dispatched)
            self.eval_queue.put_many(queue_items)

    def create_node_board(self, node_name: str, node_move: str) -> Board:
        """"""

        board: Optional[Board] = self.memory_manager.get_node_board(
            ".".join(node_name.split(".")[:-1] or "0")
        )
        if board is None:
            raise Exception(f"board not found for parent of: {node_name}")

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
