# -*- coding: utf-8 -*-

from numpy import float32

from enum import IntEnum

INF: float32 = float32(1000000.0)
DRAW: float32 = float32(0.0)
SLEEP: float32 = float32(0.01)
LOG_INTERVAL: float32 = float32(0.2)
BREAK_INTERVAL: float32 = float32(3.0)

ROOT_NODE_NAME: str = "1"
FINISHED: str = "finished"
ERROR: str = "error"


class Print(IntEnum):
    """"""

    NOTHING: int = 0
    CANDIDATES: int = 1
    TREE: int = 2
    MOVE: int = 3
    LOGS: int = 4
