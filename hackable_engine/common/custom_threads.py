# -*- coding: utf-8 -*-
from threading import Thread, Event
from typing import Any, Optional


class ReturningThread(Thread):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._return = None
        self._stop_event = Event()

    def stop(self):
        self._stop_event.set()

    def join(self, timeout: Optional[float] = None) -> Any:
        super().join(timeout)

        return self._return
