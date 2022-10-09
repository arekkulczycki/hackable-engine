"""
Custom utility thread classes.
"""

from threading import Thread
from typing import Any


class StoppableThread(Thread):
    """
    Class_docstring
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.running = True

    def stop(self) -> None:
        self.running = False


class ReturningThread(Thread):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self) -> None:
        self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args) -> Any:
        Thread.join(self, *args)
        return self._return
