# -*- coding: utf-8 -*-

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

    def _run(self):
        """

        :return:
        """

        self.setup()

        while True:
            scored_move_items = self.selector_queue.get_many(1)

            if scored_move_items:
                for scored_move_item in scored_move_items:
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

                        board = MemoryManager.get_node_board(node_name)
                        parsed_candidates = self.parse_candidates(
                            node_name, not turn_after, candidates, board
                        )

                        del self.groups[node_name]

                        self.candidates_queue.put((node_name, parsed_candidates))
            else:
                # nothing done, wait a little
                time.sleep(self.SLEEP)

    def parse_candidates(
        self, node_name: str, turn_before: bool, candidates: List[Dict], board
    ) -> List[Dict]:
        # board = MemoryManager.get_node_board(node_name)
        board.turn = turn_before  # TODO: likely useless now

        for i, candidate in enumerate(candidates):
            state = board.push_no_stack(Move.from_uci(candidate["move"]))

            candidate["i"] = i
            candidate_name = f"{node_name}.{i}"
            MemoryManager.set_node_board(candidate_name, board)

            board.light_pop(state)
        return candidates

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
