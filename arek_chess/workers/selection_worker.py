# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import time
from random import sample
from typing import Dict, List

from arek_chess.messaging import Queue
from arek_chess.workers.base_worker import BaseWorker


class SelectionWorker(BaseWorker):
    """
    Class_docstring
    """

    SLEEP = 0.001

    def __init__(self, selector_queue: Queue, candidates_queue: Queue):
        super().__init__()

        self.selector_queue = selector_queue
        self.candidates_queue = candidates_queue

        self.groups = {}

    def run(self):
        """

        :return:
        """

        while True:
            scored_move_item = self.selector_queue.get()
            if scored_move_item:
                group, size, move, fen, score = scored_move_item
                if group not in self.groups:
                    self.groups[group] = {"size": size, "moves": []}

                moves = self.groups[group]["moves"]
                moves.append({"node_name": group, "move": move, "fen": fen, "score": score})

                if len(moves) == size:
                    self.candidates_queue.put(self.select(moves))
                    del self.groups[group]
            else:
                time.sleep(self.SLEEP)

    def select(self, moves: List[Dict]) -> List[Dict]:
        """

        :return:
        """

        if len(moves) > 2:
            return sample(moves, 2)
        elif len(moves) > 0:
            return moves
        else:
            # TODO: handle error
            print("*************")
            exit(1)
