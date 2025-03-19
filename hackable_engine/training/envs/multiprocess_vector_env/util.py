from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Union

import numpy as np
from gymnasium import spaces

VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]


@dataclass
class EnvProgressData:
    time_mean: np.array
    length_mean: np.array
    return_mean: np.array
    reward_mean: np.array
    winner_mean: np.array
    time_mean: np.array
