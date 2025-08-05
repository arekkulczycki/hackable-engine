from contextlib import nullcontext

from hackable_engine.common.constants import QUEUE_HANDLER, QueueHandler

if QUEUE_HANDLER == QueueHandler.WASM:
    from threading import Lock
    from threading import Lock as LockType
else:
    from multiprocessing import Lock
    from multiprocessing.synchronize import Lock as LockType
from typing import Union, NamedTuple


class WorkerLocks(NamedTuple):
    """
    Definition of locks required for SearchWorker and DistributorWorker initialization.
    """

    status_lock: Union[LockType, nullcontext] = Lock()
    counters_lock: Union[LockType, nullcontext] = Lock()
    finish_lock: Union[LockType, nullcontext] = Lock()
    weights_lock: Union[LockType, nullcontext] = Lock()
