# -*- coding: utf-8 -*-
"""
Item passed through SelectorQueue.
"""


class SelectorItem:
    """
    Item passed through SelectorQueue.
    """

    __slots__ = (
        "node_name",
        "move_str",
        "moved_piece_type",
        "captured_piece_type",
        "score",
    )

    node_name: str
    move_str: str
    moved_piece_type: int
    captured_piece_type: int
    score: float
