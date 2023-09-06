# -*- coding: utf-8 -*-
from multiprocessing import Lock
from os import getpid
from time import sleep
from typing import Generic, List, Optional, Type, TypeVar

from arek_chess.board import GameBoardBase, GameMoveBase
from arek_chess.common.constants import (
    DISTRIBUTED,
    ROOT_NODE_NAME,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    WORKER,
)
from arek_chess.common.queue.items.control_item import ControlItem
from arek_chess.common.queue.items.distributor_item import DistributorItem
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager as QM
from arek_chess.workers.base_worker import BaseWorker

TGameBoard = TypeVar("TGameBoard", bound=GameBoardBase)
TGameMove = TypeVar("TGameMove", bound=GameMoveBase)


class DistributorWorker(BaseWorker, Generic[TGameBoard, TGameMove]):
    """
    Distributes to the queue nodes to be calculated by EvalWorkers.
    """

    status_lock: Lock
    counters_lock: Lock

    distributor_queue: QM[DistributorItem]
    eval_queue: QM[EvalItem]
    selector_queue: QM[SelectorItem]
    control_queue: QM[ControlItem]

    def __init__(
        self,
        status_lock: Lock,
        finish_lock: Lock,
        counters_lock: Lock,
        distributor_queue: QM[DistributorItem],
        eval_queue: QM[EvalItem],
        selector_queue: QM[SelectorItem],
        control_queue: QM[ControlItem],
        board_class: Type[TGameBoard],
        board_size: Optional[int],
        throttle: int,
    ):
        """"""

        super().__init__()

        self.status_lock = status_lock
        self.finish_lock = finish_lock
        self.counters_lock = counters_lock

        self.distributor_queue = distributor_queue
        self.eval_queue = eval_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue
        self.queue_throttle = throttle

        self.board: TGameBoard = board_class(size=board_size) if board_size else board_class()

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
        finished = True
        run_id: Optional[str] = None

        with self.status_lock:
            self.memory_manager.set_int(str(getpid()), 1)

        with self.counters_lock:
            memory_manager.set_int(DISTRIBUTED, 0, new=False)

        while True:
            with self.status_lock:
                status: int = memory_manager.get_int(STATUS)

            if status == Status.STARTED:

                # throttle has to be larger than value in search worker for putting
                items: List[DistributorItem] = get_items(queue_throttle * 2)

                if items:
                    # print("distrubuting items")
                    if run_id is None:
                        with self.status_lock:
                            run_id: str = memory_manager.get_str(RUN_ID)
                    distribute_items(items, run_id)

                if finished is True and items:
                    # could switch without items too, but this is for debug purposes
                    # print("distributed items: ", self.distributed, [item.node_name for item in items])
                    finished = False

                if finished:
                    print("distributor: started but no items found")
                    sleep(SLEEP * 100)

            elif not finished:
                finished = True
                run_id = None

                # empty queue, **must come before marking worker finished**
                while get_items(queue_throttle):
                    pass

                with self.counters_lock:
                    memory_manager.set_int(DISTRIBUTED, 0, new=False)

                with self.finish_lock:
                    memory_manager.set_int(f"{WORKER}_0", 1, new=False)

                self.distributed = 0

            else:
                sleep(SLEEP)

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

            # no parent will result in empty string
            if item.node_name == ROOT_NODE_NAME and len(eval_items) == 1:  # only 1 root child, so just play it
                self.control_queue.put(ControlItem(item.run_id, item.node_name))
                return

            if eval_items:
                queue_items += eval_items
            else:  # no children of this move, game over
                self.control_queue.put(ControlItem(item.run_id, item.node_name))

        if queue_items:
            self.distributed += len(queue_items)
            self.eval_queue.put_many(queue_items)

            with self.counters_lock:
                self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)

    def _get_eval_items(self, item: DistributorItem) -> List[EvalItem]:
        """
        Get all legal moves, starting from the node given in `item`, to be evaluated.
        """

        self.board.deserialize_position(item.board)

        # TODO: if it could be done efficiently, would be beneficial to check game over here

        only_forcing_moves = []
        eval_items = []
        for move in self.board.legal_moves:
            eval_item = self._get_eval_item(item, move)

            # forcing_level == -1 means that forcing moves were already taken care of before
            # forcing_level > 0 means that only forcing moves should be returned
            if item.forcing_level != 0:
                if item.forcing_level > 0 and not eval_item.forcing_level:
                    # all following moves will be of lower forcing level because of the generation order,
                    #  therefore we can break and discard the rest
                    break

                elif item.forcing_level > 0 and eval_item.forcing_level:
                    only_forcing_moves.append(eval_item)

                # if captures were analysed already then not adding to eval_items
                elif item.forcing_level == -1 and not eval_item.forcing_level:
                    eval_items.append(eval_item)

                # else pass, as in case forcing_level == -1 we discard forcing moves
            else:
                eval_items.append(eval_item)

        if item.forcing_level > 0:
            return only_forcing_moves

        return eval_items

    def _get_eval_item(self, item: DistributorItem, move: TGameMove) -> EvalItem:
        """"""

        # checking before the move is pushed, because it will change after
        forcing_level = self.board.get_forcing_level(move)

        # state = self.board.light_push(move)
        self.board.push(move)
        eval_item = EvalItem(
            item.run_id,
            item.node_name,
            move.uci(),
            forcing_level,
            self.board.serialize_position(),
        )
        # self.board.lighter_pop(state)
        self.board.pop()
        return eval_item
