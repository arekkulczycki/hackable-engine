from enum import Enum
import numpy as np


class LRShape(Enum):
    ONE = 0
    WARMUP_ONE = 1
    SIGMOID = 2
    WARMUP_SIGMOID = 3
    COSINE = 4
    COSINE_SIGMOID = 5
    WARMUP_COSINE_SIGMOID = 6
    SQUARED = 7

def get_learning_rate_decay(lr_shape, num_episodes, warm_up_len, lr_minimum_p):
    def sigmoid(episode):
        x = episode / num_episodes
        decay = -0.66 / (1 + np.e ** (-8 * (x - 0.5))) + 1
        return decay

    def warmup_sigmoid(episode):
        if episode / num_episodes < warm_up_len:
            x = episode / (warm_up_len * num_episodes)
            return (1 - lr_minimum_p) / (1 + np.e ** (-10 * (x - 0.5))) + lr_minimum_p
        x = (episode - warm_up_len * num_episodes) / ((1 - warm_up_len) * num_episodes)
        decay = -0.69 / (1 + np.e ** (-10 * (x - 0.3))) + 1.025
        return decay

    def one(episode):
        return 1

    def warmup_one(episode):
        if episode / num_episodes < warm_up_len:
            x = episode / (warm_up_len * num_episodes)
            return (1 - lr_minimum_p) / (1 + np.e ** (-10 * (x - 0.5))) + lr_minimum_p
        return 1

    def cosine(episode):
        cosine_cycle = num_episodes * warm_up_len
        cosine_step = 2 * np.pi / cosine_cycle
        return (
            np.cos(episode * cosine_step) + 1
        ) / 2.0 * (1 - lr_minimum_p) + lr_minimum_p

    def cosine_sigmoid(episode):
        return cosine(episode) * sigmoid(episode)

    def warmup_cosine_sigmoid(episode):
        if episode / num_episodes < warm_up_len:
            x = episode / (warm_up_len * num_episodes)
            return (1 - lr_minimum_p) / (1 + np.e ** (-10 * (x - 0.5))) + lr_minimum_p
        return cosine(episode) * warmup_sigmoid(episode)

    def squared(episode):
        warm_up_len = 0.2
        if episode / num_episodes < warm_up_len:
            return 1 - (num_episodes * warm_up_len - episode) / (num_episodes * warm_up_len) / 2
        # fmt: off
        return (1 - (episode - warm_up_len * num_episodes) / ((1 - warm_up_len) * num_episodes) / 3 * 2) ** 2
        # fmt: on

    lrs = {
        LRShape.SIGMOID: sigmoid,
        LRShape.WARMUP_SIGMOID: warmup_sigmoid,
        LRShape.ONE: one,
        LRShape.WARMUP_ONE: warmup_one,
        LRShape.COSINE: cosine,
        LRShape.COSINE_SIGMOID: cosine_sigmoid,
        LRShape.WARMUP_COSINE_SIGMOID: warmup_cosine_sigmoid,
        LRShape.SQUARED: squared,
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
