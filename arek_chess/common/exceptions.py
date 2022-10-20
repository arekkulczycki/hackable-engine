# -*- coding: utf-8 -*-
"""
Commonly used exception classes.
"""


class SearchFailed(Exception):
    """
    Raised when search_manager crashes and doesn't produce a move.
    """
