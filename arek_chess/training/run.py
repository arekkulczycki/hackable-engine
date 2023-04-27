import os
from argparse import ArgumentParser
from time import perf_counter

import matplotlib.pyplot as plt
import numpy
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

register(
    id="chess-v0",
    entry_point="arek_chess.training.envs.square_control_env:SquareControlEnv",
)

LOG_PATH = "./arek_chess/training/logs/"


def train(version=-1, gpu: bool = False):
    t0 = perf_counter()

    print("loading env...")
    env: DummyVecEnv = make_vec_env(
        "chess-v0", n_envs=1
    )  # , vec_env_cls=SubprocVecEnv)
    # env = make("chess-v0", fen=EQUAL_MIDDLEGAME_FEN)
    # env = VecEnv([lambda: FullBoardEnv(EQUAL_MIDDLEGAME_FEN)])
    # env = FullBoardEnv(EQUAL_MIDDLEGAME_FEN)

    env = VecMonitor(env, os.path.join(LOG_PATH, f"v{version}"))

    if version >= 0:
        # model = PPO.load(f"./chess.v{version}", env=env, custom_objects={"n_steps": 512, "learning_rate": 3e-3, "clip_range": 0.3})
        model = PPO.load(
            f"./chess.v{version}",
            env=env,
            verbose=2,
            custom_objecs={"clip_range": 0.1, "learning_rate": 3e-3, "n_steps": 512},
            device="cuda" if gpu else "auto",
        )
    else:
        print("setting up model...")
        model = PPO(
            "MlpPolicy",
            env=env,
            verbose=2,
            clip_range=0.1,
            learning_rate=3e-3,
            n_steps=512,
            device="cuda" if gpu else "auto"
        )
        # model = PPO("MultiInputPolicy", env, device="cpu", verbose=2, clip_range=0.3, learning_rate=3e-3)

    print("training started...")

    model.learn(total_timesteps=int(2**14))
    model.save(f"./chess.v{version + 1}")

    env.envs[0].controller.tear_down()
    print(f"training finished in: {perf_counter() - t0}")


def loop_train(version=-1, loops=5, gpu: bool = False):
    print("loading env...")
    env: DummyVecEnv = get_env(version)

    for _ in range(loops):
        t0 = perf_counter()
        if version >= 0:
            # custom_objects={"n_steps": 512, "learning_rate": 3e-3, "clip_range": 0.3})
            model = PPO.load(
                f"./chess.v{version}",
                env=env,
                verbose=2,
                custom_objecs={"clip_range": 0.3, "learning_rate": 3e-3, "n_steps": 256},
                device="cuda" if gpu else "cpu"
            )
        else:
            print("setting up model...")
            model = PPO(
                "MlpPolicy",
                env=env,
                verbose=2,
                clip_range=0.3,
                learning_rate=3e-3,
                n_steps=512,
                device="cuda" if gpu else "cpu"
            )
            # model = PPO("MultiInputPolicy", env, verbose=2, clip_range=0.3, learning_rate=3e-3)

        print("training started...")

        try:
            model.learn(total_timesteps=int(2**14))
        except:
            # start over with new env
            env.envs[0].controller.tear_down()
            env = get_env(version)
        else:
            # on success increment the version and keep learning
            version += 1
            model.save(f"./chess.v{version}")
        finally:
            print(f"training finished in: {perf_counter() - t0}")

    env.envs[0].controller.tear_down()


def get_env(version):
    env: DummyVecEnv = make_vec_env("chess-v0", n_envs=1)

    env = VecMonitor(env, os.path.join(LOG_PATH, f"v{version}"))

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
        "-v", "--version", type=int, default=-1, help="version of the model to use"
    )
    parser.add_argument(
        "-g", "--gpu", help="run on gpu or fail", action="store_true"
    )

    return parser.parse_args()


def find_move():
    ...


if __name__ == "__main__":
    args = get_args()
    if args.train:
        train(args.version, args.gpu)
    elif args.loop_train:
        loop_train(args.version, args.loops, args.gpu)
    elif args.plot:
        plot_results(LOG_PATH)
    else:
        find_move()
