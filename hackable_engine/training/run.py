# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from time import perf_counter

import gymnasium as gym
import numpy as np

from hackable_engine.training.algorithms import (
    cleanrl_sac,
    cleanrl_sac_logit,
)
from hackable_engine.training.device import Device
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_env import (
    MultiprocessEnv,
)
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.hyperparams import *
from hackable_engine.training.policies import policy_kwargs_map

LOG_PATH = "./hackable_engine/training/logs/"
SEARCH_LIMIT = 9
"""In the case of using tree search in training."""


def train(
    env_name: str = "default",
    version: int = -1,
    color: bool = True,
    device: Device = Device.AUTO.value,
    loops: int = 100,
):
    t0 = perf_counter()

    policy_kwargs = policy_kwargs_map[env_name]

    print("loading env...")
    if N_ENV_WORKERS == 1:
        env = get_env(
            env_name, policy_kwargs.pop("env_class"), version, color, sb3=False
        )
    else:
        env = get_multiprocess_env(
            env_name, policy_kwargs.pop("env_class"), version, color
        )

    # sb3_ppo.run(version, policy_kwargs, env, env_name, device, loops, color)
    # sb3_sac.run(version, policy_kwargs, env, env_name, device, loops, color)
    # sb3_td3.run(version, policy_kwargs, env, env_name, device, loops, color)
    # cleanrl_ppo.run(version, policy_kwargs, env, env_name, device)
    if policy_kwargs["env_type"] == "logit":
        cleanrl_sac_logit.run(version, policy_kwargs, env, env_name, device)
    else:
        cleanrl_sac.run(version, policy_kwargs, env, env_name, device)
    # cleanrl_sac_original.run(version, policy_kwargs, env, env_name, device)
    # cleanrl_td3.run(version, policy_kwargs, env, env_name, device)

    print(f"training finished in: {perf_counter() - t0}")

    if hasattr(env, "envs"):
        for e in env.envs:
            e.controller.tear_down()
            e.summarize()


def get_env(env_name, env_class, version, color, sb3: bool) -> gym.vector.VectorEnv:
    print("loading models...")
    models = [None]
    # ie = Core()
    # color_ext = "Black" if color else "White"
    # for model_version in ["A", "B", "C", "D", "E", "F"]:
    #     path = f"Hex9{color_ext}{model_version}.onnx"
    #     model = ie.compile_model(model=ie.read_model(model=path), device_name="CPU")  # GPU
    #     models.append(model)

    print("creating vec env...")
    return EpisodeStats(
        gym.make_vec(
            id=env_class.__name__,
            num_envs=N_ENVS,
            color=color,
            models=models,
            vectorization_mode=gym.VectorizeMode.SYNC,
            # vector_kwargs=dict(shared_memory=True, copy=False),
        )
    )


def get_multiprocess_env(env_name, env_class, version, color):
    # return MultiprocessAsyncEnv(
    #     lambda seed, num_envs, color_, models=[]: EpisodeStats(
    #         gym.make_vec(
    #             # id=Logit7GraphEnv.__name__,
    #             id=env_class.__name__,
    #             num_envs=num_envs,
    #             color=color_,
    #             models=models,
    #         ),
    #         is_multiprocessed=True,
    #     ),
    #     N_ENV_WORKERS,
    #     int(N_ENVS // N_ENV_WORKERS),
    #     action_shape=(1,),
    #     color=False,
    # )
    return MultiprocessEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            gym.make_vec(
                id=env_class.__name__,
                num_envs=num_envs,
                color=color_,
                models=models,
            ),
            is_multiprocessed=True,
        ),
        N_ENV_WORKERS,
        int(N_ENVS // N_ENV_WORKERS),
        color,
        action_shape=(1,),
    )


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :returns: (numpy array)
    """

    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="training mode", action="store_true")
    parser.add_argument(
        "-l", "--loops", type=int, default=100, help="iteration of training to perform"
    )
    parser.add_argument(
        "-e", "--env", help="monitor logs subpath to use for the plot", type=str
    )
    parser.add_argument(
        "-v", "--version", type=int, default=-1, help="version of the model to use"
    )
    parser.add_argument(
        "-c",
        "--color",
        type=int,
        default=1,
        help="which color player should be trained",
    )
    parser.add_argument(
        "-d",
        "--device",
        help="cuda, xpu, cpu or auto",
        choices=[d.value for d in Device.__members__.values()],
        default=Device.AUTO.value,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.train:
        train(args.env, args.version, bool(args.color), args.device, args.loops)
    else:
        print("pick a command")
