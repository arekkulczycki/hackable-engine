from enum import Enum, IntEnum

import numpy as np
# from ml_dtypes import bfloat16

FLOAT_TYPE = np.float32
INF: float = 1000000.0
INFF: FLOAT_TYPE = FLOAT_TYPE(1000000.0)
DRAW: float = 0.0
ZERO: FLOAT_TYPE = FLOAT_TYPE(0.0)
SLEEP: float = 0.001
LOG_INTERVAL: float = 3.0
BREAK_INTERVAL: float = 3.0

ROOT_NODE_NAME: str = "1"

DEBUG: str = "debug"
ACTION: str = "action"
STATUS: str = "status"
RUN_ID: str = "run_id"
WORKER: str = "worker"
DISTRIBUTED: str = "distributed"
EVALUATED: str = "evaluated"

DEFAULT_HEX_BOARD_SIZE: int = 13


class Print(IntEnum):
    """"""

    NOTHING = 0
    CANDIDATES = 1
    TREE = 2
    MOVE = 3
    LOGS = 4


PRINT_CANDIDATES = 8


class QueueHandler(IntEnum):
    FASTER_FIFO = 0
    REDIS = 1
    RABBITMQ = 2
    WASM = 3


class MemoryHandler(IntEnum):
    SHARED_MEM = 0
    REDIS = 1
    WASM = 2


class PriorityMode(IntEnum):
    FREQUENCY = 0
    MODEL = 1
    WASM = 2


class Game(str, Enum):
    CHESS = "chess"
    HEX = "hex"


class Status(IntEnum):
    STARTED = 0
    FINISHED = 1
    CLOSED = 2
    ERROR = 3


# QUEUE_HANDLER = QueueHandler.WASM
# MEMORY_HANDLER = MemoryHandler.WASM
# PRIORITY_MODE = PriorityMode.WASM
QUEUE_HANDLER = QueueHandler.FASTER_FIFO
MEMORY_HANDLER = MemoryHandler.SHARED_MEM
PRIORITY_MODE = PriorityMode.MODEL
PROCESS_COUNT = 10
QUEUE_MEMORY_MB = 100

QUEUE_THROTTLE = 64
PRINTING: Print = Print.CANDIDATES
TREE_PARAMS: str = "3,5,"
SEARCH_LIMIT: int = 13
