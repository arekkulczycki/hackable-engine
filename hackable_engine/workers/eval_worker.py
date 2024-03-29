# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
from os import cpu_count
from typing import Any, Generic, List, Optional, Tuple, TypeVar, TYPE_CHECKING

from numpy import float32

from hackable_engine.board import GameBoardBase
from hackable_engine.board.chess.chess_board import ChessBoard
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import (
    DRAW,
    INF,
    PROCESS_COUNT,
    QUEUE_THROTTLE,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    WORKER,
)
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.common.queue.items.selector_item import SelectorItem
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval
from hackable_engine.criteria.evaluation.chess.square_control_eval import (
    SquareControlEval,
)
from hackable_engine.criteria.evaluation.hex.distance_eval import DistanceEval
from hackable_engine.workers.base_worker import BaseWorker
from hackable_engine.workers.configs.eval_worker_config import EvalWorkerConfig
from hackable_engine.workers.configs.worker_locks import WorkerLocks
from hackable_engine.workers.configs.worker_queues import WorkerQueues

if TYPE_CHECKING:
    import gymnasium as gym

GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)

EVALUATORS = {
    ChessBoard: SquareControlEval(),
    HexBoard: DistanceEval(),
}


class EvalWorker(BaseWorker, Generic[GameBoardT]):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    def __init__(
        self,
        locks: WorkerLocks,
        queues: WorkerQueues,
        *,
        config: EvalWorkerConfig,
    ) -> None:
        super().__init__()

        self.locks: WorkerLocks = locks
        self.queues: WorkerQueues = queues

        self.worker_number: int = config.worker_number
        self.evaluator: Optional[BaseEval[GameBoardT]] = EVALUATORS[config.board_class]

        self.is_training_run: bool = config.is_training_run

        self.ai_model: Optional[Any] = config.ai_model
        self.ai_env: Optional[gym.Env] = None  # TODO: get from model somehow?

        self.board: GameBoardT = (
            config.board_class(size=config.board_size)
            if config.board_size
            else config.board_class()
        )

    def _set_selector_wasm_port(self, port) -> None:
        """"""

        self.queues.selector_queue.set_destination(port)

    async def _run(self) -> None:
        """"""

        # self._profile_code()
        # self.call_count = 0

        if self.pid:  # is None in WASM
            with self.locks.status_lock:
                self.memory_manager.set_int(str(self.pid), 1)

        action: Optional[WeightsType] = None
        weights_set: bool = False
        finished = True
        run_id: Optional[str] = None

        num_eval_workers = max(
            1, (PROCESS_COUNT or cpu_count()) - 2
        )  # one process required for the tree search and one for the distributor worker
        queue_throttle = QUEUE_THROTTLE // num_eval_workers

        while True:
            with self.locks.status_lock:
                status: int = self.memory_manager.get_int(STATUS)

            if status == Status.STARTED:
                finished = False

                if self.is_training_run and not weights_set:
                    # in training the action will be constant across the entire search, contrary to play analysis
                    with self.locks.weights_lock:
                        try:
                            action = self.get_memory_action(
                                self.evaluator.PARAMS_NUMBER
                            )
                        except TypeError:
                            print(
                                "failed action retrieval of size: ",
                                self.evaluator.PARAMS_NUMBER,
                            )
                    weights_set = True

                items_to_eval: List[EvalItem] = self.queues.eval_queue.get_many(
                    queue_throttle, SLEEP
                )
                if items_to_eval:
                    if run_id is None:
                        with self.locks.status_lock:
                            run_id = self.memory_manager.get_str(RUN_ID)

                    self.queues.selector_queue.put_many(
                        self.eval_items(items_to_eval, run_id, action)
                    )

                await asyncio.sleep(0)

            elif not finished:
                finished = True
                run_id = None

                weights_set = False
                with self.locks.finish_lock:
                    self.memory_manager.set_bool(
                        f"{WORKER}_{self.worker_number}", True, new=False
                    )

            else:
                await asyncio.sleep(SLEEP)

    def eval_items(
        self, eval_items: List[EvalItem], run_id: str, action: Optional[WeightsType]
    ) -> List[SelectorItem]:
        """"""

        return [
            self.eval_item(item, action) for item in eval_items if item.run_id == run_id
        ]

    def eval_item(self, item: EvalItem, action: Optional[WeightsType]) -> SelectorItem:
        """"""

        self.board.deserialize_position(item.board)

        finished = False
        result, is_check = self.get_quick_result(
            self.board, item.parent_node_name, item.move_str
        )

        if result is not None:
            finished = True
        else:
            result = self.evaluate(self.board, is_check, action)

        return SelectorItem(
            item.run_id,
            item.parent_node_name,
            item.move_str,
            -1 if finished else item.forcing_level,
            result,
            item.board,
        )

    def get_quick_result(
        self, board: GameBoardT, parent_node_name: str, move_str: str
    ) -> Tuple[Optional[float32], bool]:
        """"""

        # if board.simple_can_claim_threefold_repetition():
        if board.has_draws:
            if self.get_maybe_threefold_repetition(parent_node_name, move_str):
                return DRAW, False

        is_check = board.is_check()
        if not any(board.legal_moves):  # TODO: optimize and do on distributor?
            if (
                is_check
            ):  # TODO: currently always True for Hex, refactor in a way that makes sense
                return -INF if board.turn else INF, True

            return DRAW, True

        return None, is_check

    @staticmethod
    def get_maybe_threefold_repetition(parent_node_name: str, move_str: str) -> bool:
        """
        Identifying potential threefold repetition in an optimized way.

        WARNING: not reliable
        """

        split = f"{parent_node_name}.{move_str}".split(".")
        if len(split) < 6:
            return False

        last_6 = split[-6:]
        return (last_6[0], last_6[1]) == (last_6[4], last_6[5])

    def evaluate(
        self, board: GameBoardT, is_check: bool, weights: Optional[WeightsType]
    ) -> float32:
        """"""

        # unless constant weights are used predict weights from a given model
        if weights is None:
            weights = self.ai_model and self.get_action(board)

        return self.evaluator.get_score(board, is_check, weights=weights)

    def get_action(self, board: GameBoardT) -> WeightsType:
        """"""

        obs = self.ai_env.observation_from_board(board)
        return self.ai_model.predict(obs, deterministic=True)[
            0
        ]  # TODO: should use onnx model
