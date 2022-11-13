# -*- coding: utf-8 -*-
"""
Worker that performs evaluation on nodes picked up from queue.
"""

from collections import deque
from time import sleep
from typing import Tuple, Optional, List, Deque

from numpy import double

from arek_chess.board.board import Board
from arek_chess.common.constants import INF, SLEEP, DRAW
from arek_chess.common.memory_manager import MemoryManager
from arek_chess.common.queue_manager import QueueManager
from arek_chess.criteria.evaluation.base_eval import BaseEval
from arek_chess.criteria.evaluation.multi_idea_eval import MultiIdeaEval
from arek_chess.criteria.evaluation.square_control_eval import SquareControlEval
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
    ) -> None:
        super().__init__()

        self.input_queue: QueueManager = input_queue
        self.output_queue: QueueManager = output_queue
        self.queue_throttle: int = queue_throttle

        self.constant_action = constant_action
        self.evaluator_name = evaluator_name

    def setup(self) -> None:
        """"""

        # self.profile_code()
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
        input_queue = self.input_queue
        output_queue = self.output_queue
        queue_throttle = self.queue_throttle
        eval_items = self.eval_items
        # items_to_send: List[Tuple[str, str, int, int, double]] = []

        # switch = True
        while True:
            items_to_eval: List[Tuple[str, str]] = input_queue.get_many(queue_throttle)
            if items_to_eval:
                # put items on queue every second time
                # if switch:
                #     print(switch)
                #     output_queue.put_many([*items_to_send, *eval_items(items_to_eval, memory_manager)])
                #     items_to_send.clear()
                # else:
                #     print(switch)
                #     items_to_send = eval_items(items_to_eval, memory_manager)
                # switch = not switch

                output_queue.put_many(eval_items(items_to_eval, memory_manager))
            else:
                sleep(SLEEP)

    def run_with_buffer(self, input_queue: QueueManager, output_queue: QueueManager, queue_throttle: int):
        """
        Buffering items from queue in order to perform evaluation while waiting for new queue items.

        Offers no improvement at this moment.
        """

        get_items = self.get_items
        items_buffer = deque()
        items_to_eval: List[Tuple[str, str]] = []

        while True:
            if get_items(items_buffer, input_queue, queue_throttle):
                continue

            try:
                for _ in range(queue_throttle):
                    items_to_eval.append(items_buffer.popleft())
            except IndexError:
                pass

            if items_to_eval:
                output_queue.put_many(self.eval_items(items_to_eval, self.memory_manager))
                items_to_eval.clear()

    @staticmethod
    def get_items(buffer: Deque, input_queue: QueueManager, queue_throttle: int) -> bool:
        items_to_eval: List[Tuple[str, str]] = input_queue.get_many(queue_throttle)
        if items_to_eval:
            buffer.extend(items_to_eval)
            return True
        return False

    def eval_items(
        self, eval_items: List[Tuple[str, str]], memory_manager: MemoryManager
    ) -> List[Tuple[str, str, int, int, double]]:
        """"""

        # names = [item[0] for item in eval_items]  # generators are slower in this case :|
        # boards: List[Optional[Board]] = memory_manager.get_many_boards(names)

        # above is replaced as the parent node name is likely to repeat one after another
        boards = []
        last_name = None
        last_board = None
        for parent_node_name, move_str in eval_items:
            if parent_node_name == last_name:
                # TODO: find out why is None at times
                boards.append(last_board if last_board is not None else None)
                continue

            last_name = parent_node_name
            last_board = memory_manager.get_node_board(parent_node_name)
            boards.append(last_board)

        queue_items = [
            self.eval_item(board, node_name, move_str)
            for (node_name, move_str), board in zip(eval_items, boards)
            if board is not None
        ]

        return queue_items
        # self.output_queue.put_many(queue_items)

    def eval_item(
        self, board: Board, node_name: str, move_str: str
    ) -> Tuple[str, str, int, int, double]:
        """"""

        # self.call_count += 1

        # when reusing the same board as for the previous item, just have to revert the push done on it before
        if board is self.prev_board:
            board.lighter_pop(self.prev_state)

        captured_piece_type: int
        moved_piece_type: int
        board, captured_piece_type, moved_piece_type = self.get_board_data(
            board, move_str
        )  # board after the move
        self.prev_board = board

        result, is_check = self.get_quick_result(board, node_name, move_str)
        if result is not None:
            # sending -1 as signal game over in this node
            return node_name, move_str, moved_piece_type, -1, result

        score: double = self.evaluate(board, move_str, captured_piece_type, is_check)

        return node_name, move_str, moved_piece_type, captured_piece_type, score

    def get_quick_result(self, board: Board, node_name: str, move_str: str) -> Tuple[Optional[double], bool]:
        """"""

        # if board.simple_can_claim_threefold_repetition():
        if self.get_threefold_repetition(node_name, move_str):
            return DRAW, False

        is_check = board.is_check()
        if is_check:
            if not any(board.generate_legal_moves()):
                return -INF if board.turn else INF, True

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
    ) -> double:
        """"""

        action: Optional[BaseEval.ActionType] = (
            None
            if self.constant_action
            else self.get_action(self.evaluator.ACTION_SIZE)
        )
        return self.evaluator.get_score(
            board, move_str, captured_piece_type, is_check, action=action
        )
