# -*- coding: utf-8 -*-
from contextlib import nullcontext
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
