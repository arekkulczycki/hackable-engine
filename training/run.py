import math
from argparse import ArgumentParser
from time import perf_counter
from typing import Optional

import gym
import matplotlib.pyplot as plt
import numpy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import VecMonitor

from training.envs.base_env import BaseEnv

LOG_PATH = "./logs/"


def train(version=-1):
    t0 = perf_counter()

    env = make_vec_env("chess-v0", n_envs=4)  # , vec_env_cls=SubprocVecEnv)

    env = VecMonitor(env, LOG_PATH)

    if version >= 0:
        model = PPO.load(f"./chess.v{version}", env=env, custom_objects={"n_steps": 8, "learning_rate": 3e-5})
    else:
        model = PPO("MlpPolicy", env, verbose=2)

    model.learn(total_timesteps=int(math.pow(2, 20)))
    model.save(f"./chess.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")


def find_move(version: Optional[str] = None):
    """"""


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """

    weights = numpy.repeat(1.0, window) / window
    return numpy.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    fig = plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train", help="training mode", action="store_true")
    parser.add_argument("-l", "--plot", help="show last learning progress plot", action="store_true")
    parser.add_argument("-v", "--version", type=int, default=-1, help="version of the model to use")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.train:
        train(args.version)
    elif args.plot:
        plot_results(LOG_PATH)
    else:
        find_move()
