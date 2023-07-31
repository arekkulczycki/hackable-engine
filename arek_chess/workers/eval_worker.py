# -*- coding: utf-8 -*-
from __future__ import annotations

import signal
import typing
from typing import List, Optional, Tuple

from numpy import float32

from arek_chess.board.board import Board
from arek_chess.common.constants import (
    DRAW,
    FINISHED,
    INF,
    LOG_INTERVAL,
    RUN_ID,
    SLEEP,
    STARTED,
    STATUS,
    WORKER,
)
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import ActionType
from arek_chess.criteria.evaluation.square_control_eval import SquareControlEval
from arek_chess.workers.base_worker import BaseWorker

if typing.TYPE_CHECKING:
    import gym
    from stable_baselines3.common.base_class import BaseAlgorithmSelf


class EvalWorker(BaseWorker):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    def __init__(
        self,
        eval_queue: QueueManager,
        selector_queue: QueueManager,
        queue_throttle: int,
        worker_number: int,
        *,
        evaluator_name: Optional[str] = None,
        is_training_run: bool = False,
        env: Optional[gym.Env] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.eval_queue: QueueManager = eval_queue
        self.selector_queue: QueueManager = selector_queue
        self.queue_throttle: int = queue_throttle
        self.worker_number: int = worker_number

        self.evaluator_name = evaluator_name

        self.is_training_run: bool = is_training_run

        self.env: Optional[gym.Env] = None
        self.model: Optional[BaseAlgorithmSelf] = None
        if env:
            from stable_baselines3 import PPO

            self.env: gym.Env = env
            self.model = env and PPO.load(
                model_version,
                env=self.env,
            )

        self.board: Board = Board()

    def setup(self) -> None:
        """"""

        # self._profile_code()
        # self.call_count = 0

        # evaluators = {
        #     "optimized": OptimizedEval(),
        #     "legacy": LegacyEval(),
        #     "fast": FastEval(),
        # }

        self.evaluator = SquareControlEval()
        # self.evaluator = MultiIdeaEval()
        # self.evaluator = LegacyEval()
        # self.evaluator = FastEval()

    def _run(self) -> None:
        """"""

        self.setup()

        eval_queue = self.eval_queue
        selector_queue = self.selector_queue
        queue_throttle = self.queue_throttle
        eval_items = self.eval_items
        memory_manager = self.memory_manager

        signal.signal(signal.SIGALRM, self._handle_timeout)
        memory_manager.set_int(str(self.pid), 1)

        action: Optional[ActionType] = None
        action_set: bool = False

        while True:
            self._set_loop_timeout()

            status: int = memory_manager.get_int(STATUS)
            if status == STARTED:
                if self.is_training_run and not action_set:
                    # in training the action will be constant across the entire search, contrary to play analysis
                    action = self.get_memory_action(self.evaluator.ACTION_SIZE)
                    action_set = True

                items_to_eval: List[EvalItem] = eval_queue.get_many(
                    queue_throttle, SLEEP
                )

                if items_to_eval:
                    run_id: str = memory_manager.get_str(RUN_ID)
                    selector_queue.put_many(eval_items(items_to_eval, run_id, action))

            elif status == FINISHED:
                action_set = False
                memory_manager.set_int(f"{WORKER}_{self.worker_number}", 1, new=False)

    def _set_loop_timeout(self) -> None:
        signal.setitimer(
            signal.ITIMER_REAL, LOG_INTERVAL, 0
        )  # use constants.BREAK_INTERVAL maybe

    def _handle_timeout(self, sig, frame) -> None:
        print(f"eval timed out: {self.pid}")

    def eval_items(
        self, eval_items: List[EvalItem], run_id: str, action: Optional[ActionType]
    ) -> List[SelectorItem]:
        """"""

        return [
            self.eval_item(item, action) for item in eval_items if item.run_id == run_id
        ]

    def eval_item(self, item: EvalItem, action: Optional[ActionType]) -> SelectorItem:
        """"""

        self.board.deserialize_position(item.board)

        finished = False
        result, is_check = self.get_quick_result(
            self.board, item.node_name, item.move_str
        )
        if result is not None:
            finished = True
        else:
            result = self.evaluate(self.board, is_check, action)

        # print("eval: ", id(item.board))
        return SelectorItem(
            item.run_id,
            item.node_name,
            item.move_str,
            -1 if finished else item.captured,
            result,
            item.board,
        )

    def get_quick_result(
        self, board: Board, node_name: str, move_str: str
    ) -> Tuple[Optional[float32], bool]:
        """"""

        # if board.simple_can_claim_threefold_repetition():
        if self.get_threefold_repetition(node_name, move_str):
            return DRAW, False

        is_check = board.is_check()
        if not any(
            board.generate_legal_moves()
        ):  # TODO: optimize and do on distributor?
            if is_check:
                return -INF if board.turn else INF, True
            else:
                return DRAW, True

        return None, is_check

    @staticmethod
    def get_threefold_repetition(node_name: str, move_str: str) -> bool:
        """
        Identifying potential threefold repetition in an optimized way.

        WARNING: Not tested how reliable it is
        """

        split = node_name.split(".")
        if len(split) < 5:
            return False

        last_6 = split[-5:]
        last_6.append(move_str)
        return (last_6[0], last_6[1]) == (last_6[4], last_6[5])

    def evaluate(
        self, board: Board, is_check: bool, action: Optional[ActionType]
    ) -> float32:
        """"""

        # unless a constant action is used then predict action from the configured env
        if action is None:
            action = self.env and self.get_action()

        return self.evaluator.get_score(board, is_check, action=action)

    def get_action(self) -> ActionType:
        obs = self.env.observation()
        # return self.env.action_downgrade(self.model.predict(obs, deterministic=True)[0])
        return self.model.predict(obs, deterministic=True)[0]
