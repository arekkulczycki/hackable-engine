import sys
import time
from signal import signal, SIGTERM
from typing import Dict, List, Optional

from pyinstrument import Profiler

from arek_chess.board.board import Move, Board
from arek_chess.criteria.collection.simple_collector import SimpleCollector
from arek_chess.utils.memory_manager import MemoryManager
from arek_chess.utils.queue_manager import QueueManager
from arek_chess.workers.base_worker import BaseWorker


class CollectorWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.0005

    def __init__(self, initial_fen: Optional[str], collector_queue: QueueManager, candidates_queue: QueueManager):
        super().__init__()

        self.initial_fen = initial_fen
        self.collector_queue = collector_queue
        self.candidates_queue = candidates_queue

        self.memory_manager = MemoryManager()
        self.boards = {"0": Board(self.initial_fen) if self.initial_fen else Board()}

    def setup(self):
        # remove_shm_from_resource_tracker()

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.collector = SimpleCollector()

        self.groups = {}

        self.max_items_at_once = 1024

    def _run(self):
        """

        :return:
        """

        self.setup()

        # loop = asyncio.get_event_loop() or asyncio.new_event_loop()
        # items = loop.run_until_complete(self.get_items())

        # items = self.get_items()

        while True:
            items = self.get_items()
            # self.prioritize(items)
            self.push(items)

            # prioritize_thread = Thread(target=self.prioritize, args=(items,))
            # items_thread = ReturningThread(target=self.get_items)
            # prioritize_thread.start()
            # items_thread.start()
            # prioritize_thread.join()
            # items = items_thread.join()

            # _, items = loop.run_until_complete(
            #     asyncio.gather(self.prioritize(items), self.get_items())
            # )

    def push(self, items) -> None:
        to_put = []
        for scored_move_item in items:
            (
                node_name,
                size,
                move,
                turn_after,
                captured_piece_type,
                is_check,
                score,
            ) = scored_move_item
            to_put.append((node_name, move, score, captured_piece_type))

        self.candidates_queue.put_many(to_put)

    def prioritize(self, items) -> None:
        to_put = []
        for scored_move_item in items:
            (
                node_name,
                size,
                move,
                turn_after,
                captured_piece_type,
                is_check,
                score,
            ) = scored_move_item
            if node_name not in self.groups:
                self.groups[node_name] = {"size": size, "moves": []}

            moves = self.groups[node_name]["moves"]
            moves.append(
                {
                    "move": move,
                    "score": score,
                    "captured": captured_piece_type,
                    "is_check": is_check,
                }
            )

            if len(moves) == size:
                candidates = self.collector.order(moves, not turn_after)

                board = self.boards[node_name]

                self.store_candidates(node_name, candidates, board)  # not turn_after
                del self.groups[node_name]

                to_put.append((node_name, candidates))
                # self.candidates_queue.put((node_name, candidates))
        if to_put:
            self.candidates_queue.put_many(to_put)
        # return to_put

    def get_items(self) -> List:
        items = []
        # if to_put:
        #     await self.put_items(to_put)
        while not items:
            items = self.collector_queue.get_many(self.max_items_at_once)
            if not items:
                time.sleep(self.SLEEP)

        return items

    def store_candidates(self, node_name: str, candidates: List[Dict], board: Board) -> None:
        # board = self.memory_manager.get_node_board(node_name)
        # board.turn = turn_before  # TODO: likely useless now

        to_memo = {}
        for candidate in candidates:
            copy = board.copy()

            move = candidate["move"]
            copy.push_no_stack(Move.from_uci(move))
            # self.memory_manager.set_node_board(f"{node_name}.{move}", board)
            to_memo[f"{node_name}.{move}"] = copy

        self.memory_manager.set_many_boards(to_memo)
        self.boards.update(to_memo)

    def before_exit(self, *args):
        """"""

        if getattr(self, "should_profile_code", False):
            self.profiler.stop()
            self.profiler.print(show_all=True)

        sys.exit(0)

    def profile_code(self):
        self.profiler = Profiler()
        self.profiler.start()

        self.should_profile_code = True
