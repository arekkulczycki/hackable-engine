# -*- coding: utf-8 -*-
from typing import List, Optional

from chess import Move, KNIGHT, BISHOP

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    FINISHED,
    ROOT_NODE_NAME,
    SLEEP,
    DISTRIBUTED,
    STATUS,
    CLOSED,
)
from arek_chess.common.queue.items.control_item import ControlItem
from arek_chess.common.queue.items.distributor_item import DistributorItem
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.manager import QueueManager as QM
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

        self.board: Board = Board()

        self.node_name_cache: Optional[str] = None
        self.board_cache: Optional[Board] = None

    def setup(self):
        """"""

        self.distributed = 0

        # self._profile_code()

    def _run(self):
        """"""

        self.setup()

        memory_manager = self.memory_manager
        queue_throttle = self.queue_throttle
        get_items = self.get_items
        distribute = self.distribute
        while True:
            items: List[DistributorItem] = get_items(queue_throttle)
            if items:
                distribute(items)
            else:
                status: int = memory_manager.get_int(STATUS)
                if status == FINISHED:
                    memory_manager.set_int(STATUS, CLOSED, new=False)
                    self.distributed = 0

    def get_items(self, queue_throttle: int) -> List[DistributorItem]:
        """"""

        return self.distributor_queue.get_many(queue_throttle, SLEEP)

    def distribute(self, items: List[DistributorItem]) -> None:
        """
        Queue all legal moves for evaluation.

        :raises ValueError: when node memory was not found
        """

        queue_items = []

        for item in items:  # TODO: get_many_boards ?
            self.board.deserialize_position(item.board)

            # TODO: if it could be done efficiently, would be beneficial to check game over here
            # TODO: there is no more captured=-1, now control_queue is for special cases treatment

            recaptures = []
            new_queue_items = []
            for move in self.board.copy().legal_moves:
                eval_item = self._get_eval_item(item, move)

                if item.captured:
                    recaptured = self.board.get_captured_piece_type(move)
                    if recaptured >= item.captured or (
                        recaptured == KNIGHT and item.captured == BISHOP
                    ):
                        recaptures.append(eval_item)

                new_queue_items.append(eval_item)

            if item.captured and recaptures:
                new_queue_items = recaptures

            if item.node_name == ROOT_NODE_NAME and len(new_queue_items) == 1:
                self.control_queue.put(ControlItem(item.run_id, item.node_name))
                return

            if new_queue_items:
                queue_items += new_queue_items
            else:
                self.control_queue.put(ControlItem(item.run_id, item.node_name))

        if queue_items:
            self.distributed += len(queue_items)
            self.eval_queue.put_many(queue_items)

        if items:
            # self.control_queue.put(ControlItem(str(self.distributed)))
            try:
                self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)
            except ValueError as e:  # `Cannot mmap an empty file` randomly occurring sometimes
                # doesn't matter, will set in next iteration - probably is because SearchWorker accesses concurrently
                print(f"Setting distributed number error: {e}")

    def _get_eval_item(self, item: DistributorItem, move: Move) -> EvalItem:
        """"""

        captured = self.board.get_captured_piece_type(move)
        state = self.board.light_push(move)
        eval_item = EvalItem(
            item.run_id, item.node_name, move.uci(), captured, self.board.serialize_position()
        )
        self.board.lighter_pop(state)
        return eval_item

    def create_node_board(
        self, parent_node_name: str, node_name: str, node_move: str
    ) -> Board:
        """"""

        if parent_node_name == self.node_name_cache:
            board = self.board_cache
            board.pop()
        else:
            board: Optional[Board] = self.memory_manager.get_node_board(
                parent_node_name, self.board
            )

        if board is None:
            raise ValueError(f"board not found for parent of: {node_name}")

        self.node_name_cache = parent_node_name
        self.board_cache = board

        try:
            board.push(Move.from_uci(node_move))
        except AssertionError:
            raise ValueError(f"illegal move played")
        # print(f"setting {node_name}")
        self.memory_manager.set_node_board(node_name, board)

        return board
