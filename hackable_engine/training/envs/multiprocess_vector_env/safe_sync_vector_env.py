from gymnasium.vector import SyncVectorEnv
import numpy as np


class SafeSyncVectorEnv(SyncVectorEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._rewards = np.zeros((self.num_envs,), dtype=np.float32)
