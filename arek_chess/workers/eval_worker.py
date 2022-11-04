# -*- coding: utf-8 -*-
"""
Worker that performs evaluation on nodes picked up from queue.
"""

from time import sleep
from typing import Tuple, Optional, List

from faster_fifo import Full
from numpy import double

from arek_chess.board.board import Board
from arek_chess.common.constants import INF, SLEEP
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.queue_manager import QueueManager
from arek_chess.criteria.evaluation.optimized_eval import OptimizedEval
from arek_chess.workers.base_worker import BaseWorker


class EvalWorker(BaseWorker):
    """
    Worker that performs evaluation on nodes picked up from queue.
    """

    def __init__(
        self,
        input_queue: QueueManager,
        output_queue: QueueManager,
        queue_throttle: int,
        constant_action: bool = False,
        evaluator_name: Optional[str] = None,
    ):
        super().__init__()

        self.input_queue: QueueManager = input_queue
        self.output_queue: QueueManager = output_queue
        self.queue_throttle: int = queue_throttle

        self.constant_action = constant_action
        self.evaluator_name = evaluator_name

    def setup(self):
        """"""

        # self.profile_code()
        # self.call_count = 0

        # evaluators = {
        #     "optimized": OptimizedEval(),
        #     "legacy": LegacyEval(),
        #     "fast": FastEval(),
        # }

        self.evaluator = OptimizedEval()
        # self.evaluator = LegacyEval()
        # self.evaluator = FastEval()

    def _run(self):
        """"""

        self.setup()

        memory_manager = self.memory_manager
        input_queue = self.input_queue
        queue_throttle = self.queue_throttle
        eval_items = self.eval_items

        while True:
            items_to_eval: List[Tuple[str, str]] = input_queue.get_many(queue_throttle)
            if items_to_eval:
                eval_items(items_to_eval, memory_manager)
            else:
                sleep(SLEEP)

    def eval_items(self, eval_items: List[Tuple[str, str]], memory_manager: MemoryManager):
        """"""

        # names = [item[0] for item in eval_items]  # generators are slower in this case :|
        # boards: List[Optional[Board]] = memory_manager.get_many_boards(names)

        # above is replaced as the parent node name is likely to repeat one after another
        boards = []
        name = None
        board = None
        for item in eval_items:
            item_name = item[0]
            if item_name == name:
                # TODO: find out why is None at times
                boards.append(board.copy() if board is not None else None)
                continue

            name = item_name
            board = memory_manager.get_node_board(name)
            boards.append(board)

        queue_items = [
            self.eval_item(board, node_name, move_str)
            for (node_name, move_str), board in zip(eval_items, boards) if board is not None
        ]

        # self.output_queue.put_many(put_items)
        while True:
            try:
                self.output_queue.put_many(queue_items)
                break
            except Full:
                sleep(SLEEP)

    def eval_item(self, board: Board, node_name: str, move_str: str) -> Tuple:
        """"""

        # self.call_count += 1

        board, captured_piece_type, moved_piece_type = self.get_board_data(
            board, move_str
        )  # board after the move

        outcome = board.simple_outcome()
        if outcome is not None:
            winner = outcome.winner
            score = 0 if winner is None else INF if winner else -INF

        else:
            score = self.evaluate(board, move_str, captured_piece_type)

        return (
            node_name,
            move_str,
            moved_piece_type,
            captured_piece_type,
            score,
        )

    def evaluate(self, board, move_str, captured_piece_type) -> double:
        """"""

        action = (
            None
            if self.constant_action
            else self.get_action(self.evaluator.ACTION_SIZE)
        )
        return self.evaluator.get_score(
            board, move_str, captured_piece_type, action=action
        )
