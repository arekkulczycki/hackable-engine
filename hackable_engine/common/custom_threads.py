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


class ReturningTargetThread(ReturningThread):
    def __init__(self, target, args) -> None:
        super(ReturningThread, self).__init__(target=target, args=args)
        self._return = None
        self._stop_event = Event()

    def run(self):
        try:
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs
