"""
Constants.
"""

from numpy import double

from enum import IntEnum

INF: double = double(1000000.0)
SLEEP: double = double(0.0001)
LOG_INTERVAL: double = double(1.0)
BREAK_INTERVAL: double = double(0.3)

ROOT_NODE_NAME: str = "0"
FINISHED = "finished"


class Print(IntEnum):
    """"""

    NOTHING: int = 0
    CANDIDATES: int = 1
    TREE: int = 2
    MOVE: int = 3
