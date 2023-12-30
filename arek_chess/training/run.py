import os
import sys
from argparse import ArgumentParser
from enum import Enum
from time import perf_counter
from typing import Tuple, Union

import gym
import matplotlib.pyplot as plt
import numpy
# from gymnasium.envs.registration import register
# import sys
# import gymnasium
# sys.modules["gym"] = gymnasium
import torch
import intel_extension_for_pytorch as ipex
from onnxruntime.transformers.onnx_model_bart import BartOnnxModel
from onnxruntime.transformers.optimizer import optimize_by_fusion
from stable_baselines3 import PPO
from sbx import PPO as PPO_JAX
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from arek_chess.training.envs.hex.raw_7x7_bin_env import Raw7x7BinEnv

# sys.path.insert(0, os.getcwd())
from arek_chess.common.constants import Game, Print
from arek_chess.controller import Controller
from arek_chess.training.envs.hex.raw_5x5_bin_env import Raw5x5BinEnv
from arek_chess.training.envs.hex.simple_env import SimpleEnv
from arek_chess.training.envs.square_control_env import SquareControlEnv

# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

LOG_PATH = "./arek_chess/training/logs/"

TOTAL_TIMESTEPS = int(2**21)
LEARNING_RATE = 1e-3
N_EPOCHS = 10
N_STEPS = 2 ** 18
BATCH_SIZE = 2 ** 17  # recommended to be a factor of (N_STEPS * N_ENVS)
CLIP_RANGE = 0.1
# 1 / (1 - GAMMA) = number of steps to finish the episode, for hex env steps are SIZE^2 * SIZE^2 / 2
GAMMA = 0.99  # 0.996 for 5x5 hex, 0.999 for 7x7, 0.9997 for 9x9 - !!! values when starting on empty board !!!
GAE_LAMBDA = 0.95
ENT_COEF = 0.001  # 0.001

SEARCH_LIMIT = 9

# POLICY_KWARGS["activation_fn"] = "tanh"
policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "hex": dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
    "raw3hex": dict(net_arch=[dict(pi=[9, 9], vf=[9, 9])]),
    "raw5hex": dict(net_arch=[dict(pi=[25, 25], vf=[25, 25])]),
    "raw7hex": dict(net_arch=dict(pi=[49, 49], vf=[49, 49])),
    "raw9hex": dict(net_arch=[dict(pi=[81, 81], vf=[81, 81])]),
    "tight-fit": dict(net_arch=[dict(pi=[10, 16], vf=[16, 10])]),
    "additional-layer": dict(net_arch=[dict(pi=[10, 24, 16], vf=[16, 10])]),
}


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    XPU = "xpu"  # intel arc GPU


def train(env_name: str = "default", version: int = -1, device: Device = Device.AUTO.value):
    t0 = perf_counter()

    print("loading env...")
    # env, policy = get_env(env_name, version)
    # env, policy = get_env_hex(env_name, version)
    env, policy = get_env_hex_raw(env_name, version)

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=512, callback_after_eval=stop_train_callback, verbose=1)

    if version >= 0:
        model = PPO.load(
            f"./{env_name}.v{version}",
            env=env,
            verbose=2,
            custom_objecs={
                "clip_range": CLIP_RANGE,
                "learning_rate": LEARNING_RATE,
                "n_steps": N_STEPS,
                "n_epochs": N_EPOCHS,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "ent_coef": ENT_COEF,
            },
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=None,  #os.path.join(LOG_PATH, f"{env_name}_tensorboard")
        )
    else:
        model = PPO(
            policy,
            env=env,
            verbose=2,
            clip_range=CLIP_RANGE,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ent_coef=ENT_COEF,
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=None,  #os.path.join(LOG_PATH, f"{env_name}_tensorboard")
        )
        # model = PPO("MultiInputPolicy", env, device="cpu", verbose=2, clip_range=0.3, learning_rate=3e-3)

    print("optimizing for intel...")
    # optimizer = torch.optim.AdamW(model.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
    optimizer = torch.optim.SGD(model.policy.parameters(), lr=LEARNING_RATE, momentum=0.5)
    model.policy, optimizer = ipex.optimize(model.policy, optimizer=optimizer, dtype=torch.float32)
    # model.policy = torch.compile(model.policy, backend="ipex")

    print("training started...")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS)  # progress_bar=True
    finally:
        model.save(f"./{env_name}.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")
    # env.envs[0].summarize()
    # env.envs[0].controller.tear_down()
    env.summarize()
    env.controller.tear_down()


def loop_train(env_name: str = "default", version: int = -1, loops=5, device: Device = Device.AUTO.value):
    print("loading env...")
    # env, policy = get_env(env_name, version)
    # env, policy = get_env_hex(env_name, version)
    env, policy = get_env_hex_raw(env_name, version)

    for _ in range(loops):
        print("setting up model...")
        t0 = perf_counter()
        if version >= 0:
            # custom_objects={"n_steps": 512, "learning_rate": 3e-3, "clip_range": 0.3})
            model = PPO.load(
                f"./{env_name}.v{version}",
                env=env,
                verbose=2,
                custom_objecs={
                    "clip_range": CLIP_RANGE,
                    "learning_rate": LEARNING_RATE,
                    "n_steps": N_STEPS,
                    "n_epochs": N_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "gamma": GAMMA,
                    "gae_lambda": GAE_LAMBDA,
                    "ent_coef": ENT_COEF,
                },
                policy_kwargs=policy_kwargs_map[env_name],
                device=device,
                # tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard", f"v{version}")
            )
        else:
            model = PPO(
                policy,
                env=env,
                verbose=2,
                clip_range=CLIP_RANGE,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                ent_coef=ENT_COEF,
                policy_kwargs=policy_kwargs_map[env_name],
                device=device,
                # tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard", f"v{version}")
            )
            # model = PPO("MultiInputPolicy", env, verbose=2, clip_range=0.3, learning_rate=3e-3)

        print("training started...")

        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS)  # progress_bar=True
        except:
            print("learning crashed")
            import traceback
            traceback.print_exc()
            # start over with new env
            env.envs[0].controller.tear_down()
            env = get_env(env_name, version)
        else:
            # on success increment the version and keep learning
            version += 1

            # shouldn't be necessary but seems to be
            env.envs[0].controller.tear_down()
            env = get_env(env_name, version)
        finally:
            model.save(f"./{env_name}.v{version}")
            print(f"training finished in: {perf_counter() - t0}")

    env.envs[0].controller.tear_down()


def get_env(env_name, version) -> Tuple[gym.Env, str]:
    env: DummyVecEnv = make_vec_env(
        lambda: SquareControlEnv(
            controller=Controller(
                printing=Print.MOVE,
                search_limit=SEARCH_LIMIT,
                is_training_run=True,
                in_thread=False,
                timeout=3,
            )
        ),
        n_envs=1,
    )

    env = VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))

    return (env, "MlpPolicy")


def get_env_hex(env_name, version) -> Tuple[gym.Env, str]:
    env: DummyVecEnv = make_vec_env(
        lambda: SimpleEnv(
            controller=Controller(
                printing=Print.MOVE,
                search_limit=SEARCH_LIMIT,
                is_training_run=True,
                in_thread=False,
                timeout=3,
                game=Game.HEX,
            )
        ),
        n_envs=4,
    )

    env = VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))

    return (env, "MultiInputPolicy")


def get_env_hex_raw(env_name, version) -> Tuple[gym.Env, str]:
    env: Union[DummyVecEnv, SubprocVecEnv] = make_vec_env(
        lambda: Raw7x7BinEnv(
            controller=Controller(
                printing=Print.NOTHING,
                # tree_params="4,4,",
                search_limit=SEARCH_LIMIT,
                is_training_run=True,
                in_thread=False,
                timeout=3,
                game=Game.HEX,
                board_size=Raw7x7BinEnv.BOARD_SIZE,
            )
        ),
        # monitor_dir=os.path.join(LOG_PATH, env_name, f"v{version}"),
        n_envs=2,
        vec_env_cls=SubprocVecEnv,
        vec_env_kwargs={"start_method": "fork"},
    )

    env = VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))

    return (env, "MlpPolicy")


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :returns: (numpy array)
    """

    weights = numpy.repeat(1.0, window) / window
    return numpy.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y) :]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Rewards")
    plt.title(title + " Smoothed")
    plt.show()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="training mode", action="store_true")
    parser.add_argument(
        "-lt", "--loop-train", help="loop training mode", action="store_true"
    )
    parser.add_argument(
        "-l", "--loops", type=int, default=3, help="iteration of training to perform"
    )
    parser.add_argument(
        "-pl", "--plot", help="show last learning progress plot", action="store_true"
    )
    parser.add_argument(
        "-e", "--env", help="monitor logs subpath to use for the plot", type=str
    )
    parser.add_argument(
        "-v", "--version", type=int, default=-1, help="version of the model to use"
    )
    parser.add_argument(
        "-d",
        "--device",
        help="cuda, cpu or auto",
        choices=[d.value for d in Device.__members__.values()],
        default=Device.AUTO.value,
    )

    return parser.parse_args()


def find_move():
    ...


if __name__ == "__main__":
    args = get_args()
    if args.train:
        train(args.env, args.version, args.device)
    elif args.loop_train:
        loop_train(args.env, args.version, args.loops, args.device)
    elif args.plot:
        path = LOG_PATH if not args.env else os.path.join(LOG_PATH, args.env)
        plot_results(path)
    else:
        find_move()
