# -*- coding: utf-8 -*-

from time import perf_counter
from unittest import TestCase

from UltraDict import UltraDict
from parameterized import parameterized

from arek_chess.board.board import Board
from arek_chess.common.memory.adapters.shared_memory_adapter import SharedMemoryAdapter


class TestMemoryManager(TestCase):

    @parameterized.expand([
        [b"lalala", None, None],
        [Board().serialize_position(), None, None],
    ])
    def test_put_and_get(self, item, loader, dumper) -> None:
        sm = SharedMemoryAdapter()
        sm.set("key", item)
        assert sm.get("key") == item

        t0 = perf_counter()
        for i in range(10000):
            k = sm.get("key")
        print(perf_counter() - t0)

        ud = UltraDict()
        ud["udkey"] = item
        assert ud["udkey"] == item

        t0 = perf_counter()
        for i in range(10000):
            k = ud["udkey"]
        print(perf_counter() - t0)

        print("\n*********\n")
