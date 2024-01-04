# -*- coding: utf-8 -*-
"""
Isn't clear what envpool is supposed to improve, but here is an implemenation supported by their example.
@see https://github.com/sail-sg/envpool/blob/main/examples/sb3_examples/ppo.py

The main problem is that envs are closed before they finish.

```
envpool.register(
    "hex7-v0",
    "arek_chess.training.envs.hex.raw_7x7_bin_env",
    "Raw7x7BinEnvSpec",
    "Raw7x7BinEnvVec",
    "Raw7x7BinEnvVec",
    "Raw7x7BinEnvVec"
)
env = envpool.make("hex7-v0", env_type="gym", num_envs=8, batch_size=8)
# env.spec.id = "hex7-v0"
env = EnvpoolAdapter(env, num_envs=8)
```
"""

from typing import Optional

import numpy as np
from envpool.python.protocol import EnvPool
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs, VecEnvStepReturn


class EnvpoolAdapter(VecEnvWrapper):
    """
    Convert EnvPool object to a Stable-Baselines3 (SB3) VecEnv.
    """

    def __init__(self, venv: EnvPool, num_envs: int):
        # Retrieve the number of environments from the config
        venv.num_envs = num_envs
        super().__init__(venv=venv)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def reset(self) -> VecEnvObs:
        # if is_legacy_gym:
        return self.venv.reset()
        # else:
        #   return self.venv.reset()[0]

    def seed(self, seed: Optional[int] = None) -> None:
        # You can only seed EnvPool env by calling envpool.make()
        pass

    def step_wait(self) -> VecEnvStepReturn:
        # if is_legacy_gym:
        # TODO: this looks like a step of an already vectorized env
        obs, rewards, dones, info_dict = self.venv.step(self.actions)
        # else:
        #     obs, rewards, terms, truncs, info_dict = self.venv.step(self.actions)
        #     dones = terms + truncs
        infos = []
        # Convert dict to list of dict
        # and add terminal observation
        for i in range(self.num_envs):
            infos.append(
                {
                    key: info_dict[i][key]
                    for key in info_dict[i].keys()
                    if isinstance(info_dict[i][key], np.ndarray)
                }
            )
            if dones[i]:
                # infos[i]["terminal_observation"] = obs[i]
                # if is_legacy_gym:
                obs[i] = self.venv.reset()[i]
                # else:
                #     obs[i] = self.venv.reset(np.array([i]))[0]
        return obs, rewards, dones, infos
