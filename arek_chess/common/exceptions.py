# -*- coding: utf-8 -*-

class SearchFailed(Exception):
    """
    Raised when search crashes and doesn't produce a move.
    """


class SearchFinished(Exception):
    """
    Raised to indicate search finish.
    """
