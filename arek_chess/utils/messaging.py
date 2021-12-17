# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from queue import Empty, Full

from faster_fifo import Queue as FastQueue


class Queue:
    """
    Class_docstring
    """

    def __init__(self, name):
        """
        Initialize a queue of a chosen queuing class.
        """

        self.name = name
        self.queue = FastQueue(1000 * 1000 * 100)

    def put(self, item):
        """

        :param item:
        """

        try:
            self.queue.put(item)
        except Full:
            raise

    def get(self):
        """

        :return:
        """

        try:
            return self.queue.get(timeout=0)
        except Empty:
            return None
