# -*- coding: utf-8 -*-

from typing import Tuple, Optional, List

import gym
from numpy import float32
from stable_baselines3 import PPO

from arek_chess.board.board import Board
from arek_chess.common.constants import INF, DRAW, SLEEP
from arek_chess.common.memory.manager import MemoryManager
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue.manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import ActionType
from arek_chess.criteria.evaluation.square_control_eval import SquareControlEval
from arek_chess.workers.base_worker import BaseWorker


class EvalWorker(BaseWorker):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    def __init__(
        self,
        eval_queue: QueueManager,
        selector_queue: QueueManager,
        queue_throttle: int,
        *,
        evaluator_name: Optional[str] = None,
        memory_action: bool = False,
        env: Optional[gym.Env] = None,
        model_version: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.eval_queue: QueueManager = eval_queue
        self.selector_queue: QueueManager = selector_queue
        self.queue_throttle: int = queue_throttle

        self.evaluator_name = evaluator_name

        self.memory_action: bool = memory_action
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

        self.prev_board: Optional[Board] = None

        self.evaluator = SquareControlEval()
        # self.evaluator = MultiIdeaEval()
        # self.evaluator = LegacyEval()
        # self.evaluator = FastEval()

    def _run(self) -> None:
        """"""

        self.setup()

        memory_manager = self.memory_manager
        eval_queue = self.eval_queue
        selector_queue = self.selector_queue
        queue_throttle = self.queue_throttle
        eval_items = self.eval_items

        # switch = True
        while True:
            # items_to_eval: List[Tuple[str, str]] = input_queue.get_many_blocking(0.005, queue_throttle)
            items_to_eval: List[EvalItem] = eval_queue.get_many(queue_throttle, SLEEP)
            if items_to_eval:
                selector_queue.put_many(eval_items(items_to_eval, memory_manager))

    def eval_items(
        self, eval_items: List[EvalItem], memory_manager: MemoryManager
    ) -> List[SelectorItem]:
        """"""

        # names = [item[0] for item in eval_items]  # generators are slower in this case :|
        # boards: List[Optional[Board]] = memory_manager.get_many_boards(names)

        # above is replaced as the parent node name is likely to repeat one after another
        boards = []
        last_name = None
        last_board = None
        # for parent_node_name, move_str in (item.as_tuple() for item in eval_items):
        for item in eval_items:
            if item.node_name == last_name:
                # TODO: find out why is None at times
                boards.append(last_board if last_board is not None else None)
                continue

            last_name = item.node_name
            last_board = memory_manager.get_node_board(item.node_name)  # TODO: pass board obj for optimization
            boards.append(last_board)

        queue_items = [
            self.eval_item(board, item.node_name, item.move_str, item.run_id)
            # for (node_name, move_str), board in zip((item.as_tuple() for item in eval_items), boards)
            for item, board in zip(eval_items, boards)
            if board is not None
        ]

        return queue_items

    def eval_item(
        self, board: Board, node_name: str, move_str: str, run_id: str
    ) -> SelectorItem:
        """"""

        # self.call_count += 1

        # when reusing the same board as for the previous item, just have to revert the push done on it before
        if board is self.prev_board:
            board.lighter_pop(self.prev_state)

        captured_piece_type: int
        board, captured_piece_type = self.get_board_data(
            board, move_str
        )  # board after the move
        self.prev_board = board

        result, is_check = self.get_quick_result(board, node_name, move_str)
        if result is not None:
            # sending -1 as signal game over in this node
            return SelectorItem(run_id, node_name, move_str, result, -1)

        score: float32 = self.evaluate(board, move_str, captured_piece_type, is_check)

        return SelectorItem(run_id, node_name, move_str, score, captured_piece_type)

    def get_quick_result(
        self, board: Board, node_name: str, move_str: str
    ) -> Tuple[Optional[float32], bool]:
        """"""

        # if board.simple_can_claim_threefold_repetition():
        if self.get_threefold_repetition(node_name, move_str):
            return DRAW, False

        is_check = board.is_check()
        if not any(board.generate_legal_moves()):  # TODO: optimize and do on distributor?
            if is_check:
                return -INF if board.turn else INF, True
            else:
                return DRAW, True

        return None, is_check

    def get_threefold_repetition(self, node_name: str, move_str: str) -> bool:
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
        self, board: Board, move_str: str, captured_piece_type: int, is_check: bool
    ) -> float32:
        """"""

        action = (
            self.get_memory_action(self.evaluator.ACTION_SIZE)
            if self.memory_action
            else self.env and self.get_action()
        )

        return self.evaluator.get_score(
            board, move_str, captured_piece_type, is_check, action=action
        )

    def get_action(self) -> ActionType:
        obs = self.env.observation()
        # return self.env.action_downgrade(self.model.predict(obs, deterministic=True)[0])
        return self.model.predict(obs, deterministic=True)[0]
