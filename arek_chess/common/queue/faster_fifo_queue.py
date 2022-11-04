# -*- coding: utf-8 -*-
"""
Queue provided by external FasterFifo library.
"""

from faster_fifo import Queue

from arek_chess.common.queue.base_queue import BaseQueue


class FasterFifoQueue(BaseQueue, Queue):
    """
    Queue provided by external FasterFifo library.
    """
