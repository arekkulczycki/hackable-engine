# -*- coding: utf-8 -*-
from os import getpid
from typing import List

from chess import Move

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    DISTRIBUTED,
    FINISHED,
    ROOT_NODE_NAME,
    RUN_ID,
    SLEEP,
    STARTED,
    STATUS,
    WORKER,
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
        distribute_items = self.distribute_items

        self.memory_manager.set_int(str(getpid()), 1)

        while True:
            # throttle has to be larger than value for putting from search worker
            items: List[DistributorItem] = get_items(queue_throttle * 2)
            status: int = memory_manager.get_int(STATUS)

            if items and status == STARTED:
                run_id: str = memory_manager.get_str(RUN_ID)
                distribute_items(items, run_id)

            elif status == FINISHED:
                memory_manager.set_int(f"{WORKER}_0", 1, new=False)
                memory_manager.set_int(DISTRIBUTED, 0, new=False)
                self.distributed = 0

    def get_items(self, queue_throttle: int) -> List[DistributorItem]:
        """"""

        return self.distributor_queue.get_many(queue_throttle, SLEEP)

    def distribute_items(self, items: List[DistributorItem], run_id: str) -> None:
        """
        Queue all legal moves for evaluation.

        :raises ValueError: when node memory was not found
        """

        queue_items = []

        for item in items:
            if item.run_id != run_id:
                continue

            eval_items = self._get_eval_items(item)

            if item.node_name == ROOT_NODE_NAME and len(eval_items) == 1:
                self.control_queue.put(ControlItem(item.run_id, item.node_name))
                return

            if eval_items:
                queue_items += eval_items
            else:
                self.control_queue.put(ControlItem(item.run_id, item.node_name))

        if queue_items:
            self.distributed += len(queue_items)
            self.eval_queue.put_many(queue_items)

            self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)

    def _get_eval_items(self, item: DistributorItem) -> List[EvalItem]:
        """
        Get all legal moves, starting from the node given in `item`, to be evaluated.
        """

        self.board.deserialize_position(item.board)

        # TODO: if it could be done efficiently, would be beneficial to check game over here

        recaptures = []
        eval_items = []
        for move in self.board.legal_moves:
            eval_item = self._get_eval_item(item, move)

            # captured == -1 means that recaptures were already taken care of before
            # captured > 0 means that only recaptures should be returned
            if item.captured != 0:
                recaptured = self.board.get_captured_piece_type(move)
                if item.captured > 0 and recaptured:
                    recaptures.append(eval_item)

                # if captures were analysed already then not adding to eval_items
                if not recaptured:
                    eval_items.append(eval_item)

            else:
                eval_items.append(eval_item)

        if item.captured > 0:
            return recaptures
        return eval_items

    def _get_eval_item(self, item: DistributorItem, move: Move) -> EvalItem:
        """"""

        captured = self.board.get_captured_piece_type(move)
        # state = self.board.light_push(move)
        self.board.push(move)
        eval_item = EvalItem(
            item.run_id,
            item.node_name,
            move.uci(),
            captured,
            self.board.serialize_position(),
        )
        # self.board.lighter_pop(state)
        self.board.pop()
        return eval_item
