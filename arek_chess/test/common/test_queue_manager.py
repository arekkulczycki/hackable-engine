# -*- coding: utf-8 -*-

from time import perf_counter
from unittest import TestCase

from numpy import float32
from parameterized import parameterized

from arek_chess.common.queue.items.control_item import ControlItem
from arek_chess.common.queue.items.distributor_item import DistributorItem
from arek_chess.common.queue.items.eval_item import EvalItem
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.common.queue_manager import QueueManager


class TestQueueManager(TestCase):

    @parameterized.expand([
        ["lalala", None, None],
        [EvalItem("abc", "def"), None, None],
        [EvalItem("abc", "def"), EvalItem.loads, EvalItem.dumps],
        [ControlItem("test"), ControlItem.loads, ControlItem.dumps],
        [SelectorItem("abc", "def", float32(1.23456), 1), SelectorItem.loads, SelectorItem.dumps],
        [DistributorItem("abc", "def", float32(9.87654), 2), DistributorItem.loads, DistributorItem.dumps],
        [DistributorItem("abc", "def", float32(9.87654), 2), None, None],
    ])
    def test_put_and_get(self, item, loader, dumper) -> None:
        queue = QueueManager("test", loader=loader, dumper=dumper)
        queue.put(item)
        queue_item = queue.get()
        if hasattr(queue_item, "score"):
            assert queue_item.node_name == item.node_name
            assert queue_item.move_str == item.move_str
            assert queue_item.captured == item.captured
            assert abs(queue_item.score - item.score) < 0.00001
        else:
            assert queue_item == item
        # print(getattr(queue_item, "score", None))

        t0 = perf_counter()
        for i in range(10000):
            queue.put(item)
            queue.get()
        t = perf_counter() - t0
        print(t)
        assert t < 1
