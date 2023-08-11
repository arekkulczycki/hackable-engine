# -*- coding: utf-8 -*-

from numpy import float32

from enum import IntEnum

CPU_CORES = 8

INF: float32 = float32(1000000.0)
DRAW: float32 = float32(0.0)
SLEEP: float = 0.001
LOG_INTERVAL: float = 0.25
BREAK_INTERVAL: float = 3.0

ROOT_NODE_NAME: str = "1"

DEBUG: str = "debug"
ACTION: str = "action"
STATUS: str = "status"
RUN_ID: str = "run_id"
WORKER: str = "worker"
DISTRIBUTED: str = "distributed"

STARTED: int = 0
FINISHED: int = 1
CLOSED: int = 2
ERROR: int = 3


class Print(IntEnum):
    """"""

    NOTHING: int = 0
    CANDIDATES: int = 1
    TREE: int = 2
    MOVE: int = 3
    LOGS: int = 4


PRINT_CANDIDATES = 10


class QueueHandler(IntEnum):
    """"""

    FASTER_FIFO: int = 0
    REDIS: int = 1
    RABBITMQ: int = 2


QUEUE_HANDLER = QueueHandler.FASTER_FIFO
