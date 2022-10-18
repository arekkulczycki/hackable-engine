"""
Constants.
"""

from enum import IntEnum

INF = 1000000
SLEEP = 0.0001
LOG_INTERVAL = 1

ROOT_NODE_NAME = "0"


class Print(IntEnum):
    """"""

    NOTHING: int = 0
    CANDIDATES: int = 1
    TREE: int = 2
    BRANCH: int = 3
