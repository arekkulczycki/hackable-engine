# -*- coding: utf-8 -*-
from collections import deque
from enum import Enum

import numpy as np


class ReplayBuffer:
    class StorageType(Enum):
        LIST = 0
        DEQUE = 1

    def __init__(self, capacity, storage_type=StorageType.DEQUE):
        self.count = 0
        self.capacity = capacity
        self.storage_type = storage_type

        if storage_type is storage_type.DEQUE:
            self.buffer = deque(maxlen=capacity)
        else:
            self.buffer = []

    def push(self, position, experience):
        if self.storage_type is self.StorageType.DEQUE:
            self.buffer.append(experience)
        else:
            if self.count < self.capacity:
                self.count += 1
                self.buffer.append(experience)
            else:
                self.buffer[position] = experience

    def sample(self, size: int):
        batch_ids = np.random.randint(self.size(), size=size)
        return (self.buffer[i] for i in batch_ids)
        # return random.sample(self.buffer, size)

    def size(self):
        if self.storage_type is self.StorageType.DEQUE:
            return len(self.buffer)
        else:
            return self.count

