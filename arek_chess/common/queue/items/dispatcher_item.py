# -*- coding: utf-8 -*-
"""
Item passed through DispatcherQueue.
"""


class DispatcherItem:
    """
    Item passed through DispatcherQueue.
    """

    __slots__ = ("node_name", "move_str", "score")

    node_name: str
    move_str: str
    score: float
