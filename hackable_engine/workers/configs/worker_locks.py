# -*- coding: utf-8 -*-
from contextlib import nullcontext
from multiprocessing import Lock
from typing import Union, NamedTuple


class WorkerLocks(NamedTuple):
    """
    Definition of locks required for SearchWorker and DistributorWorker initialization.
    """

    status_lock: Union[Lock, nullcontext] = Lock()
    counters_lock: Union[Lock, nullcontext] = Lock()
    finish_lock: Union[Lock, nullcontext] = Lock()
    weights_lock: Union[Lock, nullcontext] = Lock()
