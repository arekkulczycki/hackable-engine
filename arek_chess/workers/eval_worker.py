# -*- coding: utf-8 -*-
from __future__ import annotations

import typing
from multiprocessing import Lock
from time import sleep
from typing import List, Optional, Tuple, Type, TypeVar

from numpy import float32

from arek_chess.board import GameBoardBase
from arek_chess.common.constants import (
    DRAW,
    Game, INF,
    RUN_ID,
    SLEEP,
    STATUS,
    Status,
    WORKER,
)
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import ActionType
from arek_chess.criteria.evaluation.chess.square_control_eval import SquareControlEval
from arek_chess.criteria.evaluation.hex.simple_eval import SimpleEval
from arek_chess.workers.base_worker import BaseWorker

if typing.TYPE_CHECKING:
    import gym
    from stable_baselines3.common.base_class import BaseAlgorithmSelf

TGameBoard = TypeVar("TGameBoard", bound=GameBoardBase)


class EvalWorker(BaseWorker):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    def __init__(
        self,
        status_lock: Lock,
        finish_lock: Lock,
        action_lock: Lock,
        eval_queue: QueueManager,
        selector_queue: QueueManager,
        queue_throttle: int,
        worker_number: int,
        board_class: Type[TGameBoard],
        board_size: Optional[int],
        *,
        evaluator_name: Optional[str] = None,
        is_training_run: bool = False,
        env: Optional[gym.Env] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.status_lock = status_lock
        self.finish_lock = finish_lock
        self.action_lock = action_lock

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

        self.board: TGameBoard = board_class(size=board_size) if board_size else board_class()

    def setup(self) -> None:
        """"""

        # self._profile_code()
        # self.call_count = 0

        evaluators = {
            Game.CHESS: SquareControlEval(),
            Game.HEX: SimpleEval(),
        }

        self.evaluator = evaluators[self.evaluator_name]

    def _run(self) -> None:
        """"""

        self.setup()

        eval_queue = self.eval_queue
        selector_queue = self.selector_queue
        queue_throttle = self.queue_throttle
        eval_items = self.eval_items
        memory_manager = self.memory_manager

        memory_manager.set_int(str(self.pid), 1)

        action: Optional[ActionType] = None
        action_set: bool = False
        finished = True
        run_id: Optional[str] = None

        while True:
            with self.status_lock:
                status: int = memory_manager.get_int(STATUS)

            if status == Status.STARTED:
                finished = False

                if self.is_training_run and not action_set:
                    # in training the action will be constant across the entire search, contrary to play analysis
                    with self.action_lock:
                        try:
                            action = self.get_memory_action(self.evaluator.ACTION_SIZE)
                        except TypeError:
                            print("failed action retrieval of size: ", self.evaluator.ACTION_SIZE)
                    action_set = True

                items_to_eval: List[EvalItem] = eval_queue.get_many(
                    queue_throttle, SLEEP
                )

                if items_to_eval:
                    if run_id is None:
                        with self.status_lock:
                            run_id = memory_manager.get_str(RUN_ID)

                    selector_queue.put_many(eval_items(items_to_eval, run_id, action))

            elif not finished:
                finished = True
                run_id = None

                action_set = False
                with self.finish_lock:
                    memory_manager.set_int(
                        f"{WORKER}_{self.worker_number}", 1, new=False
                    )

            else:
                sleep(SLEEP)

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
            self.board, item.parent_node_name, item.move_str
        )
        if result is not None:
            finished = True
        else:
            result = self.evaluate(self.board, is_check, action)

        # print("eval: ", id(item.board))
        return SelectorItem(
            item.run_id,
            item.parent_node_name,
            item.move_str,
            -1 if finished else item.forcing_level,
            result,
            item.board,
        )

    def get_quick_result(
        self, board: TGameBoard, parent_node_name: str, move_str: str
    ) -> Tuple[Optional[float32], bool]:
        """"""

        # if board.simple_can_claim_threefold_repetition():
        if self.get_threefold_repetition(parent_node_name, move_str):
            return DRAW, False

        is_check = board.is_check()
        if not any(board.legal_moves):  # TODO: optimize and do on distributor?
            if is_check:  # TODO: currently always True for Hex, refactor in a way that makes sense
                return -INF if board.turn else INF, True
            else:
                return DRAW, True

        return None, is_check

    @staticmethod
    def get_threefold_repetition(parent_node_name: str, move_str: str) -> bool:
        """
        Identifying potential threefold repetition in an optimized way.

        WARNING: Not tested how reliable it is
        """

        split = f"{parent_node_name}.{move_str}".split(".")
        if len(split) < 6:
            return False

        last_6 = split[-6:]
        return (last_6[0], last_6[1]) == (last_6[4], last_6[5])

    def evaluate(
        self, board: TGameBoard, is_check: bool, action: Optional[ActionType]
    ) -> float32:
        """"""

        # unless a constant action is used then predict action from the configured env
        if action is None:
            action = self.env and self.get_action(board)

        return self.evaluator.get_score(board, is_check, action=action)

    def get_action(self, board: TGameBoard) -> ActionType:
        obs = self.env.observation_from_board(board)
        # return self.env.action_downgrade(self.model.predict(obs, deterministic=True)[0])
        return self.model.predict(obs, deterministic=True)[0]
