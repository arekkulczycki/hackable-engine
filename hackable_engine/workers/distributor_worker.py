from asyncio import PriorityQueue, create_task, wait, gather, sleep
from collections import defaultdict
from random import choice
from typing import Generic, Optional, Type, TypeVar

import numpy as np

from hackable_engine.board import GameBoardBase, GameMoveBase
from hackable_engine.board.hex.move import Move
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
    SEARCH_LIMIT,
    QueueHandler,
    QUEUE_HANDLER,
    PRIORITY_MODE,
    PriorityMode,
)
from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.common.queue.items.priority_item import PriorityItem
from hackable_engine.workers.base_worker import BaseWorker
from hackable_engine.workers.configs.worker_locks import WorkerLocks
from hackable_engine.workers.configs.worker_queues import WorkerQueues
from js import ort, globalThis
from pyodide.ffi import to_js

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
        super().__init__("Distributor", memory=memory)

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

        self.move_counter = {
            True: defaultdict(lambda: 0),  # white
            False: defaultdict(lambda: 0),  # black
        }

        self.distributed = 0
        self.evaluated = 0
        self.evaluated_ratio = 0.8
        # self.received = 0
        if PRIORITY_MODE is PriorityMode.MODEL:
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session_white = ort.InferenceSession(
                f"11_white_gat.onnx", providers=["CPUExecutionProvider"], sess_options=sess_options
            )
            self.ort_session_black = ort.InferenceSession(
                f"11_black_gat.onnx", providers=["CPUExecutionProvider"], sess_options=sess_options
            )

    def _set_control_wasm_port(self, port) -> None:
        self.queues.control_queue.set_destination(port)

    def _set_eval_wasm_ports(self, ports) -> None:
        self.queues.eval_queue.set_mixed_destination(ports)

    @staticmethod
    def _run_wasm_inference(position: np.ndarray, color_to_move: bool) -> np.ndarray:
        js_input = ort.Tensor.new("float32", to_js(position.ravel()), to_js(position.shape))
        if color_to_move:
            value = globalThis.ortSessionWhite.run(input=js_input)
        else:
            value = globalThis.ortSessionBlack.run(input=js_input)
        return np.array(value["output"].data.to_py()).item()

    def _run_model_inference(self, position: np.ndarray, color_to_move: bool) -> np.ndarray:
        if color_to_move:
            return self.ort_session_white.run(None, {"inputs": position})[0]
        else:
            return self.ort_session_black.run(None, {"inputs": position})[0]

    async def _run(self) -> None:
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
        if self.distributed > 2**SEARCH_LIMIT:
            # print("distributor reached the limit")
            return
        with self.locks.counters_lock:
            self.evaluated = self.memory_manager.get_int(EVALUATED)

        size = self.priority_queue.qsize()
        if self.evaluated >= self.evaluated_ratio * self.distributed:
            if size > (1 / self.priority_get_ratio):
                to_get = int(size * self.priority_get_ratio)
            elif size >= 1:
                to_get = 1
            else:
                to_get = 0
        else:
            to_get = 0
        if to_get and size > to_get:
            items = await gather(*(self.priority_queue.get() for _ in range(to_get)))
            # print([(item.item.score, item.level) for item in items])
            await self.distribute_items(items)
        elif size >= 1:
            if not self.root_handled:
                await self.distribute_items([await self.priority_queue.get()])
                self.root_handled = True
        # else:
        #     print("empty distributor queue")

    async def put_many(self, items: list[DistributorItem]):
        for item in items:
            await self.priority_queue.put(PriorityItem(item, self.root_color))

    def get_items(self, queue_throttle: int) -> list[DistributorItem]:
        """"""

        queue_throttle = 4
        return self.queues.distributor_queue.get_many(queue_throttle, SLEEP)

    async def distribute_items(self, items: list[PriorityItem]) -> None:
        """
        Queue all legal moves for evaluation.

        :raises ValueError: when node memory was not found
        """

        queue_items = []

        white_moves_by_frequency = sorted(
            self.move_counter[True], key=lambda k: self.move_counter[True][k], reverse=True
        )
        black_moves_by_frequency = sorted(
            self.move_counter[False], key=lambda k: self.move_counter[False][k], reverse=True
        )
        for item in items:
            if item.item.run_id != self.run_id:
                continue

            node_split = item.item.node_name.split(".")
            len_node_split = len(node_split)
            item_color = self.root_color if len_node_split % 2 == 0 else not self.root_color

            # important for each item to create based on actual board state
            self.board.deserialize_position(item.item.board)

            if PRIORITY_MODE in {PriorityMode.WASM, PriorityMode.MODEL}:
                position = self.board.get_hetero_graph_node_features_one_hot()
                if PRIORITY_MODE is PriorityMode.MODEL:
                    logits = self._run_model_inference(position, self.board.turn).astype(int)
                else:
                    logits = self._run_wasm_inference(position, self.board.turn).astype(int)

                top_k = 6
                partitioned_indices = np.argpartition(logits, -top_k)[-top_k:]
                top_indices = partitioned_indices[np.argsort(logits[partitioned_indices])[::-1]]

                legal_move_set = set(self.board.legal_moves)
                eval_items = []
                for move_idx in top_indices:
                    move = Move.from_c(move_idx, self.board.size)
                    if move in legal_move_set:
                        eval_items.append(self._get_eval_item(item.item, move))
            else:
                if item.item.node_name == ROOT_NODE_NAME:
                    eval_items = self._get_eval_items(item.item)
                else:
                    eval_items = [
                        await self._get_prioritized_eval_item(
                            item, white_moves_by_frequency if item_color else black_moves_by_frequency
                        )
                    ]

            # if item.item.node_name == ROOT_NODE_NAME and len(eval_items) == 1:  # only 1 root child, so just play it
            #     self.queues.control_queue.put(ControlItem(item.item.run_id, item.item.node_name))
            #     return

            if eval_items:
                queue_items += eval_items
            else:  # no children of this move, game over
                self.queues.control_queue.put(ControlItem(item.item.run_id, item.item.node_name))

            if len_node_split > 2:  # skip the first level which gives out all legal moves
                if item_color:
                    self.move_counter[self.root_color][node_split[-1]] += 1
                else:
                    self.move_counter[not self.root_color][node_split[-1]] += 1
        if queue_items:
            self.distributed += len(queue_items)
            self.queues.eval_queue.put_many(queue_items)

            with self.locks.counters_lock:
                self.memory_manager.set_int(DISTRIBUTED, self.distributed, new=False)

    async def _get_prioritized_eval_item(self, item: PriorityItem, moves_by_frequency: list[str]) -> EvalItem:
        self.board.deserialize_position(item.item.board)
        moves_set = set(move.get_coord() for move in self.board.legal_moves)
        for move in moves_by_frequency:
            if move in moves_set and move not in item.chosen_children:
                eval_item = self._get_eval_item(item.item, Move.from_coord(move, self.board.size))
                break
        else:
            eval_item = self._get_eval_item(item.item, Move.from_coord(choice(list(moves_set)), self.board.size))

        move = eval_item.move_str
        item.add_child(move)
        if item.children_count < min(6, len(moves_set)):
            # if there are children left put back on queue
            await self.priority_queue.put(item)

        return eval_item

    def _get_eval_items(self, item: DistributorItem) -> list[EvalItem]:
        """
        Get all legal moves, starting from the node given in `item`, to be evaluated.
        """
        # TODO: if it could be done efficiently, would be beneficial to check game over here
        # parent_board_repr = self.board.as_matrix()  # .reshape(self.board.size, self.board.size)

        only_forcing_moves = []
        eval_items = []
        # board_reprs = []
        # TODO: run inference here to get only the best legal moves
        for move in self.board.legal_moves:
            eval_item = self._get_eval_item(item, move)
            # board_repr = self._board_repr_from_parent(parent_board_repr, move, self.board.turn)

            # forcing_level == -1 means that forcing moves were already taken care of before
            # forcing_level > 0 means that only forcing moves should be returned
            if item.forcing_level != 0:
                if item.forcing_level > 0 and not eval_item.forcing_level:
                    # all following moves will be of lower forcing level because of the generation order,
                    #  therefore we can break and discard the rest
                    break

                if item.forcing_level > 0 and eval_item.forcing_level:
                    only_forcing_moves.append(eval_item)
                    # board_reprs.append(board_repr)

                # if captures were analysed already then not adding to eval_items
                elif item.forcing_level == -1 and not eval_item.forcing_level:
                    eval_items.append(eval_item)
                    # board_reprs.append(board_repr)

                # else pass, as in case forcing_level == -1 we discard forcing moves
            else:
                eval_items.append(eval_item)

        if item.forcing_level > 0:
            return only_forcing_moves

        return eval_items

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
