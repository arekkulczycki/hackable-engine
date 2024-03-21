from copy import deepcopy
from typing import Callable, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn

from hackable_engine.common.queue.manager import QueueManager
from hackable_engine.training.workers.hex_board_handler_worker import HexBoardHandlerWorker


class RemoteWorkerVecEnv(DummyVecEnv):
    """"""

    def __init__(self, env_fns: List[Callable[[], gym.Env]], *, board_size: int, color: bool):
        super().__init__(env_fns)

        worker_queue = QueueManager("worker_queue")
        observation_queue = QueueManager("observation_queue")
        worker = HexBoardHandlerWorker(worker_queue, observation_queue, board_size, color, self.num_envs)
        worker.start()

    def step_wait(self) -> VecEnvStepReturn:
        # Avoid circular imports

        step_prep_data = [env.step_prep() for env in self.envs]
        for env_idx in range(self.num_envs):
            ...

        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], terminated, truncated, self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            # convert to SB3 VecEnv api
            self.buf_dones[env_idx] = terminated or truncated
            # See https://github.com/openai/gym/issues/3102
            # Gym 0.26 introduces a breaking change
            self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)

        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
