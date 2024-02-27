# -*- coding: utf-8 -*-
from dataclasses import dataclass
from time import time


@dataclass(slots=True)
class SearchWorkerCounters:  # pylint: disable=too-many-instance-attributes
    distributed: int = 0
    evaluated: int = 0
    selected: int = 0
    explored: int = 0

    last_evaluated: int = 0
    last_distributed: int = 0
    last_external_distributed: int = 0

    time: float = time()
