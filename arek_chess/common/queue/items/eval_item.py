# -*- coding: utf-8 -*-
"""
Item passed through EvalQueue.
"""


class EvalItem:
    """
    Item passed through EvalQueue.
    """

    __slots__ = ("node_name", "move_str")

    node_name: str
    move_str: str
