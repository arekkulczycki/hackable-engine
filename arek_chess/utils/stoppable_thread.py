"""
Module_docstring.
"""

from threading import Thread


class StoppableThread(Thread):
    """
    Class_docstring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running = True

    def stop(self):
        self.running = False
