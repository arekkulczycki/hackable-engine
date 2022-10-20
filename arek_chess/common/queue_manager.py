"""
Module_docstring.
"""

from functools import partial
from queue import Empty, Full

from faster_fifo import Queue

from larch.pickle.pickle import dumps, loads


class QueueManager:
    """
    Class_docstring
    """

    def __init__(self, name):
        """
        Initialize a queue of a chosen queuing class.
        """

        self.name = name

        queue_loads = lambda _bytes: loads(_bytes.tobytes())
        queue_dumps = partial(dumps, protocol=5, with_refs=False)
        self.queue = Queue(
            max_size_bytes=1000 * 1000 * 100,
            loads=queue_loads,
            dumps=queue_dumps,
        )

    def put(self, item):
        """

        :param item:
        """

        try:
            self.queue.put(item)
        except Full:
            raise

    def put_many(self, items):
        """

        :param items:
        """

        try:
            self.queue.put_many_nowait(items)
        except Full:
            raise

    def get(self):
        """

        :return:
        """

        try:
            return self.queue.get_nowait(timeout=0)
        except Empty:
            return None

    def get_many(self, max_messages_to_get: int = 10):
        """

        :return:
        """

        try:
            return self.queue.get_many_nowait(max_messages_to_get=max_messages_to_get)
        except Empty:
            return None

    def empty(self) -> bool:
        """

        :return:
        """

        return self.queue.empty()

    def size(self) -> int:
        """

        :return:
        """

        return self.queue.qsize()
