# -*- coding: utf-8 -*-


class BaseQueue:
    """
    Base class for a multiprocessing queue.
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
