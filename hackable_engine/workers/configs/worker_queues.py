# -*- coding: utf-8 -*-
from typing import NamedTuple

from hackable_engine.common.queue.items.control_item import ControlItem
from hackable_engine.common.queue.items.distributor_item import DistributorItem
from hackable_engine.common.queue.items.eval_item import EvalItem
from hackable_engine.common.queue.items.selector_item import SelectorItem
from hackable_engine.common.queue.manager import QueueManager as QM


class WorkerQueues(NamedTuple):
    """
    Definition of queues required for worker initialization.
    """

    distributor_queue: QM[DistributorItem]
    eval_queue: QM[EvalItem]
    selector_queue: QM[SelectorItem]
    control_queue: QM[ControlItem]
