import os
from argparse import ArgumentParser
from enum import Enum
from time import perf_counter

import matplotlib.pyplot as plt
import numpy
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

from arek_chess.common.constants import Print
from arek_chess.controller import Controller
from arek_chess.training.envs.square_control_env import SquareControlEnv

# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

register(
    id="chess-v0",
    entry_point="arek_chess.training.envs.square_control_env:SquareControlEnv",
)

LOG_PATH = "./arek_chess/training/logs/"

TOTAL_TIMESTEPS = int(2**13)  # keeps failing before finish on 2**14
LEARNING_RATE = 1e-3
N_EPOCHS = 10
N_STEPS = 512
BATCH_SIZE = 128  # recommended to be a factor of (N_STEPS * N_ENVS)
CLIP_RANGE = 0.3

SEARCH_LIMIT = 9

# POLICY_KWARGS = dict(net_arch=[dict(pi=[10, 24, 16], vf=[16, 10])])
# POLICY_KWARGS["activation_fn"] = "tanh"
policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "tight-fit": dict(net_arch=[dict(pi=[10, 16], vf=[16, 10])]),
    "additional-layer": dict(net_arch=[dict(pi=[10, 24, 16], vf=[16, 10])]),
}


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"


def train(env_name: str = "default", version: int = -1, device: Device = Device.AUTO.value):
    t0 = perf_counter()

    print("loading env...")
    env: DummyVecEnv = get_env(env_name, version)

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=512, callback_after_eval=stop_train_callback, verbose=1)

    if version >= 0:
        # model = PPO.load(f"./chess.v{version}", env=env, custom_objects={"n_steps": 512, "learning_rate": 3e-3, "clip_range": 0.3})
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
            },
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
        )
    else:
        print("setting up model...")
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=2,
            clip_range=CLIP_RANGE,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
        )
        # model = PPO("MultiInputPolicy", env, device="cpu", verbose=2, clip_range=0.3, learning_rate=3e-3)

    print("training started...")

    model.learn(total_timesteps=TOTAL_TIMESTEPS)  # progress_bar=True
    model.save(f"./{env_name}.v{version + 1}")

    env.envs[0].controller.tear_down()
    print(f"training finished in: {perf_counter() - t0}")


def loop_train(env_name: str = "default", version: int = -1, loops=5, device: Device = Device.AUTO.value):
    print("loading env...")
    env: DummyVecEnv = get_env(env_name, version)

    for _ in range(loops):
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
                },
                policy_kwargs=policy_kwargs_map[env_name],
                device=device,
            )
        else:
            print("setting up model...")
            model = PPO(
                "MlpPolicy",
                env=env,
                verbose=2,
                clip_range=CLIP_RANGE,
                learning_rate=LEARNING_RATE,
                n_steps=N_STEPS,
                n_epochs=N_EPOCHS,
                batch_size=BATCH_SIZE,
                policy_kwargs=policy_kwargs_map[env_name],
                device=device,
            )
            # model = PPO("MultiInputPolicy", env, verbose=2, clip_range=0.3, learning_rate=3e-3)

        print("training started...")

        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS)  # progress_bar=True
        except:
            # start over with new env
            env.envs[0].controller.tear_down()
            env = get_env(env_name, version)
        else:
            # on success increment the version and keep learning
            version += 1
            model.save(f"./{env_name}.v{version}")
        finally:
            print(f"training finished in: {perf_counter() - t0}")

    env.envs[0].controller.tear_down()


def get_env(env_name, version):
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

    return env


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
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
