"""
Custom utility thread classes.
"""

from threading import Thread, Event
from typing import Any


class ReturningThread(Thread):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._return = None
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def join(self, timeout: float = None) -> Any:
        Thread.join(self, timeout)

        return self._return
