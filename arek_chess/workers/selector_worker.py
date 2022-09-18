import asyncio
import os
import time
from signal import signal, SIGTERM
from typing import Dict, List, Tuple

from pyinstrument import Profiler

from arek_chess.board.board import Move, Board
from arek_chess.criteria.pre_selection.legacy_selector import LegacySelector
from arek_chess.utils.memory_manager import (
    MemoryManager,
)
from arek_chess.utils.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker


class SelectorWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.0005

    def __init__(self, selector_queue: Queue, candidates_queue: Queue):
        super().__init__()

        self.memory_manager = MemoryManager()
        self.selector_queue = selector_queue
        self.candidates_queue = candidates_queue

        self.boards = {}

    def setup(self):
        # remove_shm_from_resource_tracker()

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.selector = LegacySelector()

        self.groups = {}

        self.max_items_at_once = 1024

    def _run(self):
        """

        :return:
        """

        self.setup()

        loop = asyncio.get_event_loop() or asyncio.new_event_loop()

        items = loop.run_until_complete(self.get_items())

        while True:
            _, items = loop.run_until_complete(
                asyncio.gather(self.select(items), self.get_items())
            )

    async def select(self, items) -> None:
        to_put = []
        for scored_move_item in items:
            (
                node_name,
                size,
                move,
                turn_after,
                captured_piece_type,
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
                }
            )

            if len(moves) == size:
                candidates = self.selector.select(moves, not turn_after)

                try:
                    board = self.boards[node_name]
                except KeyError:
                    print("selector", f"Not found: {node_name}")
                    board = Board()
                    for move in node_name.split(".")[1:]:
                        board.push(Move.from_uci(move))

                await self.store_candidates(node_name, candidates, board)  # not turn_after
                del self.groups[node_name]

                to_put.append((node_name, candidates))
                # self.candidates_queue.put((node_name, candidates))
        if to_put:
            await self.put_items(to_put)
        # return to_put

    async def get_items(self) -> List:
        items = []
        # if to_put:
        #     await self.put_items(to_put)
        while not items:
            items = self.selector_queue.get_many(self.max_items_at_once)
        return items

    async def put_items(self, items):
        self.candidates_queue.put_many(items)

    async def store_candidates(self, node_name: str, candidates: List[Dict], board: Board) -> None:
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

        exit(0)

    def profile_code(self):
        self.profiler = Profiler()
        self.profiler.start()

        self.should_profile_code = True
