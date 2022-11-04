# -*- coding: utf-8 -*-
"""
Queue provided by external FasterFifo library.

pip install rq==1.11.*
"""

from typing import Tuple, List

from arek_chess.common.queue.base_queue import BaseQueue

from rq import Queue


class RedisQueue(BaseQueue, Queue):
    """
    Queue provided by external FasterFifo library.
    """

    def __init__(self, name):
        """"""

        super().__init__(name, job_class=tuple)

    def get_many(self, number_to_get: int = 10) -> Tuple:
        """"""

        return self.get_jobs(length=number_to_get)

    def put_many(self, jobs: List[Tuple]) -> None:
        """"""

        self.enqueue_many(jobs)
