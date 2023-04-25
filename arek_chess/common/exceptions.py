# -*- coding: utf-8 -*-
"""
Commonly used exception classes.
"""


class SearchFailed(Exception):
    """
    Raised when search crashes and doesn't produce a move.
    """


class SearchFinished(Exception):
    """
    Raised to indicate search finish.
    """
