# -*- coding: utf-8 -*-
from enum import Enum, IntEnum

import numpy as np
import torch as th
# from ml_dtypes import bfloat16

FLOAT_TYPE = np.float32
TH_FLOAT_TYPE = th.float32
INF: float = 1000000.0
DRAW: float = 0.0
ZERO: float = 0.0
SLEEP: float = 0.001
LOG_INTERVAL: float = 0.5
BREAK_INTERVAL: float = 3.0

ROOT_NODE_NAME: str = "1"

DEBUG: str = "debug"
ACTION: str = "action"
STATUS: str = "status"
RUN_ID: str = "run_id"
WORKER: str = "worker"
DISTRIBUTED: str = "distributed"

DEFAULT_HEX_BOARD_SIZE: int = 13


class Print(IntEnum):
    """"""

    NOTHING: int = 0
    CANDIDATES: int = 1
    TREE: int = 2
    MOVE: int = 3
    LOGS: int = 4


PRINT_CANDIDATES = 5


class QueueHandler(IntEnum):
    """"""

    FASTER_FIFO: int = 0
    REDIS: int = 1
    RABBITMQ: int = 2
    WASM: int = 3


class MemoryHandler(IntEnum):
    """"""

    SHARED_MEM: int = 0
    REDIS: int = 1
    WASM: int = 2


class Game(str, Enum):
    """"""

    CHESS: str = "chess"
    HEX: str = "hex"


class Status(IntEnum):
    STARTED: int = 0
    FINISHED: int = 1
    CLOSED: int = 2
    ERROR: int = 3


# QUEUE_HANDLER = QueueHandler.WASM
# MEMORY_HANDLER = MemoryHandler.WASM
QUEUE_HANDLER = QueueHandler.FASTER_FIFO
MEMORY_HANDLER = MemoryHandler.SHARED_MEM
PROCESS_COUNT = 10
QUEUE_MEMORY_MB = 100

QUEUE_THROTTLE = 64
PRINTING: Print = Print.CANDIDATES
TREE_PARAMS: str = "3,5,"
SEARCH_LIMIT: int = 14
