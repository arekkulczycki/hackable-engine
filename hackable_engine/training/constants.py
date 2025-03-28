from enum import Enum
import numpy as np


class LRShape(Enum):
    ONE = 0
    REVERSE_SIGMOID = 1
    WARMUP_SIGMOID = 2
    SQUARED = 3

def get_learning_rate_decay(lr_shape, num_episodes):
    def reverse_sigmoid(episode):
        x = episode / num_episodes
        decay = -0.66 / (1 + np.e ** (-6 * (x - 0.5))) + 1
        return decay

    def warmup_sigmoid(episode):
        warm_up = 0.12
        if episode / num_episodes < warm_up:
            x = episode / (warm_up * num_episodes)
            return 0.99 / (1 + np.e ** (-10 * (x - 0.5))) + 0.01
        x = (episode - warm_up * num_episodes) / ((1 - warm_up) * num_episodes)
        decay = -0.69 / (1 + np.e ** (-10 * (x - 0.3))) + 1.025
        return decay

    def squared(episode):
        warm_up = 0.2
        if episode / num_episodes < warm_up:
            return 1 - (num_episodes * warm_up - episode) / (num_episodes * warm_up) / 2
        # fmt: off
        return (1 - (episode-warm_up*num_episodes) / ((1-warm_up)*num_episodes) / 3 * 2) ** 2
        # fmt: on

    def one(episode):
        return 1
    lrs = {
        LRShape.REVERSE_SIGMOID: reverse_sigmoid,
        LRShape.WARMUP_SIGMOID: warmup_sigmoid,
        LRShape.SQUARED: squared,
        LRShape.ONE: one,
    }

    return lrs[lr_shape]