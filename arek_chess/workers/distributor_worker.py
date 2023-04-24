# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional

from chess import Move, KNIGHT, BISHOP
from numpy import float32

from arek_chess.board.board import Board
from arek_chess.common.constants import FINISHED, ROOT_NODE_NAME, ERROR, SLEEP
from arek_chess.common.queue_manager import QueueManager as QM
from arek_chess.workers.base_worker import BaseWorker


class DistributorWorker(BaseWorker):
    """
    Distributes to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(
        self,
        distributor_queue: QM,
        eval_queue: QM,
        selector_queue: QM,
        control_queue: QM,
        throttle: int,
    ):
        """"""

        super().__init__()

        self.distributor_queue = distributor_queue
        self.eval_queue = eval_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = throttle

        self.node_name_cache: Optional[str] = None
        self.board_cache: Optional[Board] = None

    def setup(self):
        """"""

        self.distributed = 0

        # self._profile_code()

    def _run(self):
        """"""

        self.setup()

        queue_throttle = self.queue_throttle
        get_items = self.get_items
        distribute = self.distribute
        while True:
            items: List[Tuple[str, str, float32, int]] = get_items(queue_throttle)
            if items:
                distribute(items)

    def get_items(self, queue_throttle: int) -> List[Tuple[str, str, float32, int]]:
        """"""

        return self.distributor_queue.get_many(queue_throttle, SLEEP)

    def distribute(self, items: List[Tuple[str, str, float32, int]]) -> None:
        """
        Queue all legal moves for evaluation.

        :raises ValueError: when node memory was not found
        """

        queue_items = []

        for node_name, move_str, score, captured in items:  # TODO: get_many_boards ?
            # not a real node, just a signal for finishing processing iteration
            if node_name == FINISHED:
                self.control_queue.put(str(self.distributed))
                self.control_queue.put(FINISHED)
                self.distributed = 0
                return None

            # root board is already created at the very start in the search_manager
            if node_name == ROOT_NODE_NAME:
                board = self.memory_manager.get_node_board(node_name)
                if board is None:
                    raise ValueError(f"node memory not found: {node_name}")
            else:
                parent_node_name = ".".join(node_name.split(".")[:-1])

                # storing the shared memory for node_name board
                try:
                    board = self.create_node_board(
                        parent_node_name, node_name, move_str
                    )
                except ValueError as e:
                    # FIXME: still getting items from previous run...
                    self.control_queue.put(ERROR)
                    break

            new_queue_items = []
            for move in board.legal_moves:
                if captured:
                    recaptured = board.get_captured_piece_type(move)
                    if recaptured >= captured or (recaptured == KNIGHT and captured == BISHOP):
                        new_queue_items.append((node_name, move.uci()))
                else:
                    new_queue_items.append((node_name, move.uci()))

            # TODO: if it could be done efficiently, would be beneficial to check game over here
            # new_queue_items = []
            # move: Move
            # for move in board.legal_moves:
            #     outcome = self.get_outcome(board, move)
            #
            #     if outcome is not None:
            #         winner: bool = outcome.winner
            #         score = 0 if winner is None else INF if winner else -INF
            #
            #         # if game over then return the move back to the selector
            #         self.selector_queue.put(
            #             (
            #                 parent_node_name,  # will always be assigned because root position is never game over
            #                 move_str,
            #                 0,
            #                 -1,  # sending -1 as signal game over in this node
            #                 score,
            #             )
            #         )
            #     else:
            #         new_queue_items.append((node_name, move.uci()))

            if new_queue_items:
                queue_items += new_queue_items
            else:
                self.control_queue.put(node_name)

        if queue_items:
            self.distributed += len(queue_items)
            self.control_queue.put(str(self.distributed))
            self.eval_queue.put_many(queue_items)

    def create_node_board(
        self, parent_node_name: str, node_name: str, node_move: str
    ) -> Board:
        """"""

        if parent_node_name == self.node_name_cache:
            board = self.board_cache
            board.pop()
        else:
            board: Optional[Board] = self.memory_manager.get_node_board(parent_node_name)

        if board is None:
            raise ValueError(f"board not found for parent of: {node_name}")

        self.node_name_cache = parent_node_name
        self.board_cache = board

        board.push(Move.from_uci(node_move))
        self.memory_manager.set_node_board(node_name, board)

        return board
