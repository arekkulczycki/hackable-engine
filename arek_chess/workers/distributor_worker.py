# -*- coding: utf-8 -*-
import asyncio
import contextlib
from multiprocessing import Lock
from typing import Any, Generic, List, Optional, Type, TypeVar

from arek_chess.board import GameBoardBase, GameMoveBase
from arek_chess.common.constants import (
    DISTRIBUTED,
    QUEUE_THROTTLE, ROOT_NODE_NAME,
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
        memory: Optional[Any] = None
    ):
        """"""

        super().__init__(memory)

        self.status_lock = status_lock or contextlib.nullcontext()
        self.finish_lock = finish_lock or contextlib.nullcontext()
        self.counters_lock = counters_lock or contextlib.nullcontext()

        self.distributor_queue = distributor_queue
        self.eval_queue = eval_queue
        self.selector_queue = selector_queue
        self.control_queue = control_queue

        self.board: TGameBoard = board_class(size=board_size) if board_size else board_class()

    def _set_control_wasm_port(self, port) -> None:
        """"""

        self.control_queue.set_destination(port)

    def _set_eval_wasm_ports(self, ports) -> None:
        """"""

        self.eval_queue.set_mixed_destination(ports)

    def setup(self):
        """"""

        self.distributed = 0

        # self._profile_code()

    async def _run(self):
        """"""

        self.setup()

        get_items = self.get_items
        finished = True
        run_id: Optional[str] = None

        if self.pid:  # is None in WASM
            with self.status_lock:
                self.memory_manager.set_int(str(self.pid), 1)

        with self.counters_lock:
            self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

        while True:
            with self.status_lock:
                status: int = self.memory_manager.get_int(STATUS)

            if status == Status.STARTED:
                # throttle has to be larger than value in search worker for putting
                items: List[DistributorItem] = get_items(QUEUE_THROTTLE * 2)

                if items:
                    if run_id is None:
                        with self.status_lock:
                            run_id: str = self.memory_manager.get_str(RUN_ID)
                    self.distribute_items(items, run_id)

                if finished is True and items:
                    # could switch without items too, but this is for debug purposes
                    # print("distributed items: ", self.distributed, [item.node_name for item in items])
                    finished = False

                if finished:
                    # print("distributor: started but no items found")
                    await asyncio.sleep(SLEEP * 100)
                else:
                    await asyncio.sleep(0)

            elif not finished:
                finished = True
                run_id = None

                # empty queue, **must come before marking worker finished**
                while get_items(QUEUE_THROTTLE):
                    pass

                with self.counters_lock:
                    self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

                with self.finish_lock:
                    self.memory_manager.set_bool(f"{WORKER}_0", True, new=False)

                self.distributed = 0

            else:
                await asyncio.sleep(SLEEP)

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
