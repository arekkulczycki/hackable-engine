# -*- coding: utf-8 -*-
import asyncio
from typing import cast, Generic, List, Optional, Type, TypeVar

import numpy as np
from nptyping import NDArray

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
)
from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.eval_item import EvalItem
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
        board_class: Type[GameBoardT],
        board_size: Optional[int],
    ):
        super().__init__()

        self.locks: WorkerLocks = locks
        self.queues: WorkerQueues = queues

        self.board: GameBoardT = (
            board_class(size=board_size) if board_size else board_class()
        )

        self.distributed = 0

    def _set_control_wasm_port(self, port) -> None:
        """"""

        self.queues.control_queue.set_destination(port)

    def _set_eval_wasm_ports(self, ports) -> None:
        """"""

        self.queues.eval_queue.set_mixed_destination(ports)

    async def _run(self) -> None:
        """"""

        # self._profile_code()

        get_items = self.get_items
        finished = True
        run_id: Optional[str] = None

        if self.pid:  # is None in WASM
            with self.locks.status_lock:
                self.memory_manager.set_int(str(self.pid), 1)

        with self.locks.counters_lock:
            self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

        while True:
            with self.locks.status_lock:
                status: int = self.memory_manager.get_int(STATUS)

            if status == Status.STARTED:
                # throttle has to be larger than value in search worker for putting
                items: List[DistributorItem] = get_items(QUEUE_THROTTLE * 2)

                if items:
                    if run_id is None:
                        with self.locks.status_lock:
                            run_id = self.memory_manager.get_str(RUN_ID)
                    self.distribute_items(items, cast(str, run_id))

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

                with self.locks.counters_lock:
                    self.memory_manager.set_int(DISTRIBUTED, 0, new=False)

                with self.locks.finish_lock:
                    self.memory_manager.set_bool(f"{WORKER}_0", True, new=False)

                self.distributed = 0

            else:
                await asyncio.sleep(SLEEP)

    def get_items(self, queue_throttle: int) -> List[DistributorItem]:
        """"""

        return self.queues.distributor_queue.get_many(queue_throttle, SLEEP)

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

            if (
                item.node_name == ROOT_NODE_NAME and len(eval_items) == 1
            ):  # only 1 root child, so just play it
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

    def _get_eval_items(self, item: DistributorItem) -> List[EvalItem]:
        """
        Get all legal moves, starting from the node given in `item`, to be evaluated.
        """

        self.board.deserialize_position(item.board)

        # TODO: if it could be done efficiently, would be beneficial to check game over here

        parent_board_repr = self.board.as_matrix().reshape(self.board.size, self.board.size)

        only_forcing_moves = []
        eval_items = []
        board_reprs = []
        for move in self.board.legal_moves:
            eval_item = self._get_eval_item(item, move)
            board_repr = self._board_repr_from_parent(parent_board_repr, move)

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

    def _get_eval_scores(self, board_matrices: List[NDArray]) -> List[np.float32]:
        """"""

        return [
            ZERO for _ in range(len(board_matrices))
        ]  # TODO: initialize and use a model
        # return self.model.run(None, {"inputs": np.stack(np.asarray(board_matrices), axis=0)})[0][0]

    @staticmethod
    def _items_with_scores(
        items: List[EvalItem], scores: List[np.float32]
    ) -> List[EvalItem]:
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
    def _board_repr_from_parent(parent_board_repr: NDArray, move: GameMoveT) -> NDArray:
        """"""

        board_repr = parent_board_repr.copy()
        board_repr[move.x][move.y] = 1
        return board_repr
