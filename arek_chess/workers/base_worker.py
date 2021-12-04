# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from multiprocessing import Process


class BaseWorker(Process):
    """
    Class_docstring
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
