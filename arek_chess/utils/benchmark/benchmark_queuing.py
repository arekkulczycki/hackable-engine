from time import time

from arek_chess.utils.messaging import Queue

BENCHMARK = {
    "bytes": bin(242543),
    "short string": "test",
    "long string": "test my long string\ntest my long string\ntest my long string\ntest my long string\n"
                   "test my long string\ntest my long string\ntest my long string\ntest my long string\n"
                   "test my long string\ntest my long string\ntest my long string\ntest my long string\n",
    "single type tuple": ("abc", "def", "ghi"),
    "multi type tuple": ("abc", 12, 0.536324132),
    "dict": {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"},
    "tuple with dict": ("abc", 123, {"abc": 0.564562, "def": 0.225342, "ghi": 34262745, "jkl": "string"}),
}

queue = Queue("benchmark")


def repeat(f, value):
    t0 = time()
    for i in range(10000):
        f(value)
    return time() - t0


def put_and_get(value):
    queue.put(value)
    queue_value = queue.get()
    assert queue_value == value


for k, v in BENCHMARK.items():
    t = repeat(put_and_get, v)
    print(f"{k}: {t}")
