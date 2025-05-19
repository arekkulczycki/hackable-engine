from enum import Enum
import numpy as np


class LRShape(Enum):
    ONE = 0
    REVERSE_SIGMOID = 1
    WARMUP_SIGMOID = 2
    WARMUP_ONE = 3
    SQUARED = 4

def get_learning_rate_decay(lr_shape, num_episodes, warm_up_len):
    def reverse_sigmoid(episode):
        x = episode / num_episodes
        decay = -0.66 / (1 + np.e ** (-8 * (x - 0.5))) + 1
        return decay

    def warmup_sigmoid(episode):
        if episode / num_episodes < warm_up_len:
            x = episode / (warm_up_len * num_episodes)
            return 0.99 / (1 + np.e ** (-10 * (x - 0.5))) + 0.01
        x = (episode - warm_up_len * num_episodes) / ((1 - warm_up_len) * num_episodes)
        decay = -0.69 / (1 + np.e ** (-10 * (x - 0.3))) + 1.025
        return decay

    def warmup_one(episode):
        if episode / num_episodes < warm_up_len:
            x = episode / (warm_up_len * num_episodes)
            return 0.99 / (1 + np.e ** (-10 * (x - 0.5))) + 0.01
        return 1

    def squared(episode):
        warm_up_len = 0.2
        if episode / num_episodes < warm_up_len:
            return 1 - (num_episodes * warm_up_len - episode) / (num_episodes * warm_up_len) / 2
        # fmt: off
        return (1 - (episode - warm_up_len * num_episodes) / ((1 - warm_up_len) * num_episodes) / 3 * 2) ** 2
        # fmt: on

    def one(episode):
        return 1
    lrs = {
        LRShape.REVERSE_SIGMOID: reverse_sigmoid,
        LRShape.WARMUP_SIGMOID: warmup_sigmoid,
        LRShape.WARMUP_ONE: warmup_one,
        LRShape.SQUARED: squared,
        LRShape.ONE: one,
    }

    return lrs[lr_shape]


class TargetUpdateMode(Enum):
    HARD = 0
    SOFT = 1


class GammaMode(Enum):
    MANUAL = 0
    TRAINED = 1
    RETRAINED = 2


class BufferMode(Enum):
    RAM = 0
    DISK = 1
    RAM_AND_DISK = 2
