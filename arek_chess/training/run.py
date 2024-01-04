import os
from argparse import ArgumentParser
from enum import Enum
from time import perf_counter
from typing import Tuple, Union

import gym
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import numpy

# from gymnasium.envs.registration import register
# import sys
# import gymnasium
# sys.modules["gym"] = gymnasium
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

# sys.path.insert(0, os.getcwd())
from arek_chess.common.constants import Game, Print
from arek_chess.controller import Controller
from arek_chess.training.envs.hex.raw_7_env import Raw7Env
from arek_chess.training.envs.hex.raw_7x7_bin_env import Raw7x7BinEnv
from arek_chess.training.envs.hex.simple_env import SimpleEnv
from arek_chess.training.envs.square_control_env import SquareControlEnv
from arek_chess.training.hex_cnn_features_extractor import HexCnnFeaturesExtractor

# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

LOG_PATH = "./arek_chess/training/logs/"

N_ENVS = 2**3
TOTAL_TIMESTEPS = int(2**22)
LEARNING_RATE = 1e-3
N_EPOCHS = 10
N_STEPS = 2**11  # batch size per env, total batch size is this times N_ENVS
BATCH_SIZE = int(N_STEPS * N_ENVS / 2**0)   # mini-batch, recommended to be a factor of (N_STEPS * N_ENVS)
CLIP_RANGE = 0.5  # 0.1 to 0.3 according to many sources, but I used even 0.9 with beneficial results
# 1 / (1 - GAMMA) = number of steps to finish the episode, for hex env steps are SIZE^2 * SIZE^2 / 2
GAMMA = 0.99  # 0.996 for 5x5 hex, 0.999 for 7x7, 0.9997 for 9x9 - !!! values when starting on empty board !!!
GAE_LAMBDA = 0.95
ENT_COEF = 0.001  # 0 to 0.01 https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe

SEARCH_LIMIT = 9

# POLICY_KWARGS["activation_fn"] = "tanh"
policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "hex": dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
    "raw3hex": dict(net_arch=[dict(pi=[9, 9], vf=[9, 9])]),
    "raw5hex": dict(net_arch=[dict(pi=[25, 25], vf=[25, 25])]),
    "raw7hex": dict(net_arch=dict(pi=[49, 49], vf=[49, 49])),
    "raw7hexcnn": dict(
        policy="CnnPolicy",
        features_extractor_class=HexCnnFeaturesExtractor,
        features_extractor_kwargs=dict(n_envs=N_ENVS, board_size=7, features_dim=49, kernel_size=5),
        # should_preprocess_obs=False,
        net_arch=[49, 49],
    ),
    "raw7hex-v2": dict(net_arch=dict(pi=[49, 49], vf=[49, 49])),
    "raw9hex": dict(net_arch=[dict(pi=[81, 81], vf=[81, 81])]),
    "tight-fit": dict(net_arch=[dict(pi=[10, 16], vf=[16, 10])]),
    "additional-layer": dict(net_arch=[dict(pi=[10, 24, 16], vf=[16, 10])]),
}


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    XPU = "xpu"  # intel Arc GPU


def train(
    env_name: str = "default", version: int = -1, device: Device = Device.AUTO.value
):
    t0 = perf_counter()

    print("loading env...")
    env = get_env(env_name, Raw7Env, version, device)

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=512, callback_after_eval=stop_train_callback, verbose=1)

    if version >= 0:
        policy_kwargs_map[env_name].pop("policy", "MlpPolicy")
        model = PPO.load(
            f"./{env_name}.v{version}",
            env=env,
            verbose=2,
            param_override={
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
            tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )
    else:
        model = PPO(
            policy_kwargs_map[env_name].pop("policy", "MlpPolicy"),
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
            tensorboard_log=None,  # os.path.join(LOG_PATH, f"{env_name}_tensorboard")
        )

    print("optimizing for intel...")
    # optimizer = torch.optim.AdamW(model.policy.parameters(), lr=LEARNING_RATE, eps=1e-5)
    optimizer = torch.optim.SGD(
        model.policy.parameters(), lr=LEARNING_RATE, momentum=0.9
    )
    if device == Device.XPU and hasattr(torch, "xpu") and torch.xpu.is_available():
        model.policy, model.optimizer = torch.xpu.optimize(
            model.policy,
            optimizer=optimizer,
            # dtype=torch.bfloat16,
            weights_prepack=False,
            optimize_lstm=True,
            # fuse_update_step=True,
            # auto_kernel_selection=True,
        )
    else:
        model.policy, model.optimizer = ipex.optimize(
            model.policy,
            optimizer=optimizer,
            # dtype=torch.bfloat16,
            weights_prepack=False,
            optimize_lstm=True,
            # fuse_update_step=True,
            # auto_kernel_selection=True,
        )
    # model.policy = torch.compile(model.policy, backend="ipex")

    # print("optimizing with lightning")
    # fabric = lightning.Fabric(accelerator="cpu", devices=4, strategy="ddp")
    # fabric.launch()
    # model.policy, optimizer = fabric.setup(model.policy, optimizer)

    print("training started...")
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
    finally:
        model.save(f"./{env_name}.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")
    # env.envs[0].summarize()
    # env.envs[0].controller.tear_down()
    # env.summarize()
    for e in env.envs:
        e.controller.tear_down()


def loop_train(
    env_name: str = "default",
    version: int = -1,
    loops=5,
    device: Device = Device.AUTO.value,
):
    print("loading env...")
    env = get_env(env_name, Raw7x7BinEnv, version, device)

    for _ in range(loops):
        print("setting up model...")
        t0 = perf_counter()
        if version >= 0:
            policy_kwargs_map[env_name].pop("policy", "MlpPolicy"),
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
                policy_kwargs_map[env_name].pop("policy", "MlpPolicy"),
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
            model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
        except:
            print("learning crashed")
            import traceback

            traceback.print_exc()
            # start over with new env
            for e in env.envs:
                e.controller.tear_down()
            env = get_env(env_name, Raw7x7BinEnv, version, device)
        else:
            # on success increment the version and keep learning
            version += 1

            # shouldn't be necessary but seems to be
            for e in env.envs:
                e.controller.tear_down()
            env = get_env(env_name, Raw7x7BinEnv, version, device)
        finally:
            model.save(f"./{env_name}.v{version}")
            print(f"training finished in: {perf_counter() - t0}")

    for e in env.envs:
        e.controller.tear_down()


def get_env(env_name, env_class, version, device) -> gym.Env:
    env: Union[DummyVecEnv, SubprocVecEnv] = make_vec_env(
        lambda: env_class(),
        # monitor_dir=os.path.join(LOG_PATH, env_name, f"v{version}"),
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        # vec_env_kwargs={"start_method": "fork"},
    )
    env.device = device

    # nice try, but works only with Box and Dict spaces
    # env = VecFrameStack(env, n_stack=8)

    return VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))


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
    y = moving_average(y, window=250)
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
        help="cuda, xpu, cpu or auto",
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
