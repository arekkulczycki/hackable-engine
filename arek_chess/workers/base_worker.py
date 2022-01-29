# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from multiprocessing import Process


class BaseWorker(Process):
    """
    Base for the worker process.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self):
        try:
            self._run()
        except KeyboardInterrupt:
            print("fuck it")
            exit(0)

    def _run(self):
        raise NotImplementedError
