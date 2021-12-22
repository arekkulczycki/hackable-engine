# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from signal import signal, SIGTERM
from typing import Dict, List

from pyinstrument import Profiler

from arek_chess.board.board import Move
from arek_chess.criteria.pre_selection.legacy_selector import LegacySelector
from arek_chess.utils.memory_manager import (
    MemoryManager,
    remove_shm_from_resource_tracker,
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

        self.selector_queue = selector_queue
        self.candidates_queue = candidates_queue

    def setup(self):
        remove_shm_from_resource_tracker()

        signal(SIGTERM, self.before_exit)

        # self.profile_code()

        self.selector = LegacySelector()

        self.groups = {}

        self.hanging_board_memory = set()

    def run(self):
        """

        :return:
        """

        self.setup()

        while True:
            scored_move_item = self.selector_queue.get()

            if scored_move_item:
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
                    self.hanging_board_memory.add(node_name)

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

                    board = MemoryManager.get_node_board(node_name)
                    self.candidates_queue.put(
                        (
                            node_name,
                            self.parse_candidates(
                                node_name, not turn_after, candidates, board
                            ),
                        )
                    )

                    del self.groups[node_name]

                    # erase memory for params and board for the parent node of all candidates
                    MemoryManager.remove_node_memory(node_name)
                    try:
                        self.hanging_board_memory.remove(node_name)
                    except KeyError:
                        continue

            else:
                # nothing done, wait a little
                time.sleep(self.SLEEP)

    def parse_candidates(
        self, node_name: str, turn_before: bool, candidates: List[Dict], board
    ) -> List[Dict]:
        # board = MemoryManager.get_node_board(node_name)
        board.turn = turn_before

        for i, candidate in enumerate(candidates):
            state = board.push_no_stack(Move.from_uci(candidate["move"]))

            candidate["i"] = i
            candidate_name = f"{node_name}.{i}"
            MemoryManager.set_node_board(candidate_name, board)

            self.hanging_board_memory.add(candidate_name)

            board.light_pop(state)
        return candidates

    def before_exit(self, *args):
        """"""

        print("cleaning...")

        for node_name in self.hanging_board_memory:
            try:
                MemoryManager.remove_node_board_memory(node_name)
            except:
                print(f"error erasing: {node_name}")
                continue

        print("cleaning done")

        if getattr(self, "should_profile_code", False):
            self.profiler.stop()
            self.profiler.print(show_all=True)

        exit(0)

    def profile_code(self):
        self.profiler = Profiler()
        self.profiler.start()

        self.should_profile_code = True
