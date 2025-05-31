# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Union

import numpy as np

VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]


@dataclass
class EnvProgressData:
    time_mean: np.array
    length_mean: np.array
    win_length_mean: np.array
    loss_length_mean: np.array
    return_mean: np.array
    reward_mean: np.array
    winner_mean: np.array
    legal_mean: np.array
    time_mean: np.array
