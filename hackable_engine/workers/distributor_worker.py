# -*- coding: utf-8 -*-
from asyncio import PriorityQueue, create_task, wait, gather, sleep
from typing import Generic, Optional, Type, TypeVar

import numpy as np

from hackable_engine.board import GameBoardBase, GameMoveBase
from hackable_engine.common.constants import (
    DISTRIBUTED,
    QUEUE_THROTTLE,
    ROOT_NODE_NAME,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    WORKER,
    ZERO,
    EVALUATED,
)
from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.common.queue.items.priority_item import PriorityItem
from hackable_engine.workers.base_worker import BaseWorker
from hackable_engine.workers.configs.worker_locks import WorkerLocks
from hackable_engine.workers.configs.worker_queues import WorkerQueues

GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)
GameMoveT = TypeVar("GameMoveT", bound=GameMoveBase)


class DistributorWorker(BaseWorker, Generic[GameBoardT, GameMoveT]):
    """
    Distributes to the queue nodes to be calculated by EvalWorkers.
    """

    def __init__(
        self,
        locks: WorkerLocks,
        queues: WorkerQueues,
        board_class: Type[GameBoardT],  # cannot pass board because is not picklable for a new process
        board_size: Optional[int],
        root_color: bool,
        memory=None,
    ):
        super().__init__(memory)

        self.locks: WorkerLocks = locks
        self.queues: WorkerQueues = queues
        # TODO: consider janus?
        self.priority_queue_size = 100_000
        self.priority_get_ratio = 0.001
        self.priority_get_size = int(self.priority_queue_size * self.priority_get_ratio)
        self.priority_queue: PriorityQueue = PriorityQueue(maxsize=self.priority_queue_size + self.priority_get_size)

        self.board: GameBoardT = board_class(size=board_size) if board_size else board_class()
        self.root_color = root_color
        """if white, then looking for the best black move, then looking for the lowest score"""

        self.distributed = 0
        self.evaluated = 0
        self.evaluated_ratio = 0.8
        # self.received = 0

    def _set_control_wasm_port(self, port) -> None:
        """"""

        self.queues.control_queue.set_destination(port)

    def _set_eval_wasm_ports(self, ports) -> None:
        """"""

        self.queues.eval_queue.set_mixed_destination(ports)

    async def _run(self) -> None:
        """"""

        # self._profile_code()
        self.root_handled = False
        self.run_id: Optional[str] = None

        if self.pid:  # is None in WASM
            with self.locks.status_lock:
                self.memory_manager.set_int(str(self.pid), 1)

        with self.locks.counters_lock:
            self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

        while True:
            with self.locks.status_lock:
                status: int = self.memory_manager.get_int(STATUS)

            if status == Status.STARTED:
                if self.run_id is None:
                    with self.locks.status_lock:
                        self.run_id = self.memory_manager.get_str(RUN_ID).replace("\x00", "")

                await self.prioritize_distribution()

            elif self.run_id is not None:
                self.root_handled = False
                self.run_id = None

                # empty queue, **must come before marking worker finished**
                while self.get_items(QUEUE_THROTTLE):
                    pass

                with self.locks.counters_lock:
                    self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

                with self.locks.finish_lock:
                    self.memory_manager.set_bool(f"{WORKER}_0", True, new=False)

                self.distributed = 0

            else:
                await sleep(SLEEP)

    async def prioritize_distribution(self) -> None:
        # throttle has to be larger than value in search worker for putting
        items: list[DistributorItem] = self.get_items(QUEUE_THROTTLE * 2)
        # self.received += len(items)

        await wait((create_task(self.put_many(items)), create_task(self.distribute_prioritized())))

    async def distribute_prioritized(self):
        with self.locks.counters_lock:
            self.evaluated = self.memory_manager.get_int(EVALUATED)

        size = self.priority_queue.qsize()
        if self.evaluated >= self.evaluated_ratio * self.distributed:
            if size > (1 / self.priority_get_ratio):
                to_get = int(size * self.priority_get_ratio)
            else:
                to_get = 1
        else:
            to_get = 0
        if to_get:
            items = await gather(*(self.priority_queue.get() for _ in range(to_get)))
            # print([(item.item.score, item.level) for item in items])
            self.distribute_items([item.item for item in items])
        elif size >= 1:
            if not self.root_handled:
                self.distribute_items([(await self.priority_queue.get()).item])
                self.root_handled = True

    async def put_many(self, items: list[DistributorItem]):
        for item in items:
            await self.priority_queue.put(PriorityItem(item, self.root_color))

    def get_items(self, queue_throttle: int) -> list[DistributorItem]:
        """"""

        queue_throttle = 4
        return self.queues.distributor_queue.get_many(queue_throttle, SLEEP)

    def distribute_items(self, items: list[DistributorItem]) -> None:
        """
        Queue all legal moves for evaluation.

        :raises ValueError: when node memory was not found
        """

        queue_items = []

        for item in items:
            if item.run_id != self.run_id:
                continue

            eval_items = self._get_eval_items(item)

            if item.node_name == ROOT_NODE_NAME and len(eval_items) == 1:  # only 1 root child, so just play it
                self.queues.control_queue.put(ControlItem(item.run_id, item.node_name))
                return

            if eval_items:
                queue_items += eval_items
            else:  # no children of this move, game over
                self.queues.control_queue.put(ControlItem(item.run_id, item.node_name))

        if queue_items:
            self.distributed += len(queue_items)
            self.queues.eval_queue.put_many(queue_items)

            with self.locks.counters_lock:
                self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)

    def _get_eval_items(self, item: DistributorItem) -> list[EvalItem]:
        """
        Get all legal moves, starting from the node given in `item`, to be evaluated.
        """

        self.board.deserialize_position(item.board)

        # TODO: if it could be done efficiently, would be beneficial to check game over here

        parent_board_repr = self.board.as_matrix()  # .reshape(self.board.size, self.board.size)

        only_forcing_moves = []
        eval_items = []
        board_reprs = []
        # TODO: drop _get_eval_scores later and run inference here to get only the best legal moves
        for move in self.board.legal_moves:
            eval_item = self._get_eval_item(item, move)
            board_repr = self._board_repr_from_parent(parent_board_repr, move, self.board.turn)

            # forcing_level == -1 means that forcing moves were already taken care of before
            # forcing_level > 0 means that only forcing moves should be returned
            if item.forcing_level != 0:
                if item.forcing_level > 0 and not eval_item.forcing_level:
                    # all following moves will be of lower forcing level because of the generation order,
                    #  therefore we can break and discard the rest
                    break

                if item.forcing_level > 0 and eval_item.forcing_level:
                    only_forcing_moves.append(eval_item)
                    board_reprs.append(board_repr)

                # if captures were analysed already then not adding to eval_items
                elif item.forcing_level == -1 and not eval_item.forcing_level:
                    eval_items.append(eval_item)
                    board_reprs.append(board_repr)

                # else pass, as in case forcing_level == -1 we discard forcing moves
            else:
                eval_items.append(eval_item)

        scores = self._get_eval_scores(board_reprs)

        if item.forcing_level > 0:
            return self._items_with_scores(only_forcing_moves, scores)

        return self._items_with_scores(eval_items, scores)

    def _get_eval_scores(self, board_matrices: list[np.ndarray]) -> list[np.float32]:
        """"""

        return [ZERO for _ in range(len(board_matrices))]  # TODO: initialize and use a model
        # return self.model.run(None, {"inputs": np.stack(np.asarray(board_matrices), axis=0)})[0][0]

    @staticmethod
    def _items_with_scores(items: list[EvalItem], scores: list[np.float32]) -> list[EvalItem]:
        """"""

        for eval_item, score in zip(items, scores):
            eval_item.model_score = score
        return items

    def _get_eval_item(self, item: DistributorItem, move: GameMoveT) -> EvalItem:
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
            ZERO,
            self.board.serialize_position(),
        )
        self.board.pop()
        return eval_item

    @staticmethod
    def _board_repr_from_parent(parent_board_repr: np.ndarray, move: GameMoveT, color: bool) -> np.ndarray:
        """board_repr array is (3, size, size), first dim is (empty, white, black)"""

        idx = 1 if color else 2
        board_repr = parent_board_repr.copy()
        board_repr[idx][move.x][move.y] = 1
        return board_repr
