# -*- coding: utf-8 -*-
from typing import Optional, List, Union, Sequence, Any

import numpy as np
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.vector.vector_env import ArrayType, VectorEnv
from ray.rllib import BaseEnv

from hackable_engine.training.envs.multiprocess_vector_env.ray_remote_env import RayRemoteEnv


class RayVectorEnv(VectorEnv):

    def __init__(self, make_env, num_workers, per_worker_env, color):
        self.n_envs = num_workers * per_worker_env

        self.one_env = make_env(0, per_worker_env, color)
        self.single_observation_space = self.one_env.single_observation_space
        self.single_action_space = self.one_env.single_action_space
        self.remote: BaseEnv = RayRemoteEnv(
            make_env, num_workers, per_worker_env, False, color
        )
        super().__init__()

    @property
    def length_queue(self):
        # TODO: get from child envs
        return self.remote.length_queue

    @property
    def return_queue(self):
        # TODO: get from child envs
        return self.remote.return_queue

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:  # type: ignore
        return self.remote.poll()[0], {}

    def step(
        self, actions: ActType
    ) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        self.remote.send_actions(actions)
        return self.remote.poll()

    # def step_async(self, actions: np.ndarray) -> None:
    #     self.remote.send_actions(actions)
    #
    # def step_wait(self) -> VecEnvStepReturn:
    #     return self.remote.poll()

    def render(self) -> tuple[RenderFrame, ...] | None:
        return self.remote.render()

    def close(self) -> None:
        self.remote.stop()

    def get_attr(self, name: str) -> tuple[Any, ...]:
        attr = getattr(self.one_env, name)
        return tuple(attr for _ in range(self.n_envs))

    def set_attr(self, name: str, values: list[Any] | tuple[Any] | object):
        pass

    # def env_method(
    #     self,
    #     method_name: str,
    #     *method_args,
    #     indices: VecEnvIndices = None,
    #     **method_kwargs,
    # ) -> List[Any]:
    #     pass
    #
    # def env_is_wrapped(
    #     self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    # ) -> List[bool]:
    #     return [True for _ in range(self.n_envs)]

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass
