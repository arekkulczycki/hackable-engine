import heapq
from collections import deque

from sortedcontainers import SortedList

from hackable_engine.common.constants import FLOAT_TYPE


class RealTimeMedian:
    def __init__(self):
        self.low: list[FLOAT_TYPE] = []  # max heap (inverted min-heap)
        self.high: list[FLOAT_TYPE] = []  # min heap

    def insert(self, num):
        # Add to max heap
        heapq.heappush(self.low, -num)

        # Balance step: move the largest in low to high
        heapq.heappush(self.high, -heapq.heappop(self.low))

        # Maintain size property: low can have one more element than high
        if len(self.low) < len(self.high):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def get_median(self):
        if len(self.low) > len(self.high):
            return -self.low[0]
        else:
            return (-self.low[0] + self.high[0]) / 2


class BoundedRealTimeMedian:
    def __init__(self, *, max_size: int):
        self.max_size: int = max_size
        self.window: deque = deque()
        self.sorted: SortedList = SortedList()

        self.count: int = 0

    def insert(self, value):
        self.window.append(value)
        self.sorted.add(value)

        if self.count > self.max_size:
            old = self.window.popleft()
            self.sorted.remove(old)
        else:
            self.count += 1

    def get_median(self):
        n = len(self.sorted)
        if n == 0:
            return None
        if n % 2 == 1:
            return self.sorted[n // 2]
        else:
            return (self.sorted[n // 2 - 1] + self.sorted[n // 2]) / 2


class RealTimeMeanVariance:
    def __init__(self):
        self.count: int = 1
        self.mean: float = 0.0
        self.M2: float = 0.0  # sum of squares of differences from the current mean

    def update(self, x: FLOAT_TYPE):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def get_mean(self):
        return self.mean

    def get_variance(self):
        return self.M2 / self.count

    def get_std(self):
        return (self.get_variance()) ** 0.5

    def is_within_std(self, value: FLOAT_TYPE) -> bool:
        # values are always > 0
        return value <= self.mean + self.get_std()

    def is_within_half_std(self, value: FLOAT_TYPE) -> bool:
        return value <= self.mean + self.get_std() / 2
