from dataclasses import dataclass
from time import time

from arek_chess.utils.queue_manager import QueueManager


class PicklableClass:
    __slots__ = ("val1", "val2", "val3", "val4", "val5")

    def __init__(self, *args):
        self.val1, self.val2, self.val3, self.val4, self.val5 = args

    def __eq__(self, other):
        return all([getattr(self, _attr) == getattr(other, _attr) for _attr in ["val1", "val2", "val3", "val4", "val5"]])


@dataclass
class Dataclass:
    val1: str
    val2: str
    val3: int
    val4: float
    val5: bool


# from arek_chess.utils.benchmark.dataklass import dataklass
# @dataklass
# class Dataklass:
#     val1: str
#     val2: str
#     val3: int
#     val4: float
#     val5: bool


BENCHMARK = {
    "bytes": bin(242543),
    "short string": "test",
    "long string": "test my long string\ntest my long string\ntest my long string\ntest my long string\n"
                   "test my long string\ntest my long string\ntest my long string\ntest my long string\n"
                   "test my long string\ntest my long string\ntest my long string\ntest my long string\n",
    "single type tuple": ("abc", "def", "ghi"),
    "multi type tuple": ("abc", 12, 0.536324132),
    "dict": {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
    "list of dicts": [
        {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
        {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
        {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
        {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
    ],
    "tuple with dict": ("abc", 123, {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"}),
    "picklable_class": PicklableClass("test1", "test2", 123, 123.123, False),
    "dataclass": Dataclass("test1", "test2", 123, 123.123, False),
    # "dataklass": Dataklass("test1", "test2", 123, 123.123, False),
}

queue = Queue("benchmark")


def repeat(f, value):
    t0 = time()
    for i in range(10000):
        f(value)
    return time() - t0


def put_and_get(value, _print=False):
    queue.put(value)
    queue_value = queue.get()

    if _print:
        print(queue_value, value)

    assert queue_value == value


for k, v in BENCHMARK.items():
    t = repeat(put_and_get, v)
    print(f"{k}: {t}")
