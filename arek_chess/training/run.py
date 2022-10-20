import math
from argparse import ArgumentParser
from time import perf_counter

import matplotlib.pyplot as plt
import numpy
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import VecMonitor

register(
    id='chess-v0',
    entry_point='arek_chess.training.envs:FullBoardEnv',
)

LOG_PATH = "./logs/"
EQUAL_MIDDLEGAME_FEN = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"


def train(version=-1):
    t0 = perf_counter()

    env = make_vec_env("chess-v0", n_envs=1, env_kwargs={"fen": EQUAL_MIDDLEGAME_FEN})  # , vec_env_cls=SubprocVecEnv)
    # env = make("chess-v0", fen=EQUAL_MIDDLEGAME_FEN)
    # env = VecEnv([lambda: FullBoardEnv(EQUAL_MIDDLEGAME_FEN)])
    # env = FullBoardEnv(EQUAL_MIDDLEGAME_FEN)

    env = VecMonitor(env, LOG_PATH)

    if version >= 0:
        model = PPO.load(f"./chess.v{version}", env=env)
    else:
        model = PPO("MlpPolicy", env, verbose=2, n_steps=64)

    print("training started...")

    model.learn(total_timesteps=int(math.pow(2, 0)))
    model.save(f"./chess.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")


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


def find_move():
    ...


if __name__ == "__main__":
    args = get_args()
    if args.train:
        train(args.version)
    elif args.plot:
        plot_results(LOG_PATH)
    else:
        find_move()
