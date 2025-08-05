from typing import NamedTuple

import numpy as np


class Chunks(NamedTuple):
    buffer_indices_chunks: list[np.ndarray]
    states_chunks: tuple
    next_states_chunks: tuple
    actions_chunks: tuple
    rewards_chunks: tuple
    dones_chunks: tuple
    control_stats_chunks: tuple
    target_q_values_chunks: tuple
