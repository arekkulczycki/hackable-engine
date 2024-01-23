import os
from argparse import ArgumentParser
from enum import Enum
from multiprocessing import set_start_method
from time import perf_counter
from typing import Union
import multiprocessing as mp
import gym
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch as th
from RayEnvWrapper import WrapperRayVecEnv
from stable_baselines3 import PPO
#from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)

from arek_chess.training.policies import policy_kwargs_map

# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

LOG_PATH = "./arek_chess/training/logs/"

N_ENVS = 2**12
N_ENV_WORKERS = 10
TOTAL_TIMESTEPS = int(2**27)
LEARNING_RATE = 1e-2  # lambda p: max(1e-2 * p**3, 1e-6)
N_EPOCHS = 10
N_STEPS = 2**7  # batch size per env, total batch size is this times N_ENVS
BATCH_SIZE = int(N_STEPS * N_ENVS / 2**3)   # mini-batch, recommended to be a factor of (N_STEPS * N_ENVS)
CLIP_RANGE = 0.2  # 0.1 to 0.3 according to many sources, but I used even 0.9 with beneficial results
# 1 / (1 - GAMMA) = number of steps to finish the episode, for hex env steps are SIZE^2 * SIZE^2 / 2
GAMMA = 0.9995  # 0.996 for 5x5 hex, 0.999 for 7x7, 0.9997 for 9x9 - !!! values when starting on empty board !!!
GAE_LAMBDA = 0.95  # 0.95 to 0.99, controls the trade-off between bias and variance
ENT_COEF = 0.3  # 0 to 0.01 https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
SGD_MOMENTUM = 0.5
ADAMW_WEIGHT_DECAY = 1e-2

SEARCH_LIMIT = 9


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    XPU = "xpu"  # intel Arc GPU


def train(
    env_name: str = "default", version: int = -1, color: bool = True, device: Device = Device.AUTO.value
):
    t0 = perf_counter()

    policy_kwargs = policy_kwargs_map[env_name]

    print("loading env...")
    # env = get_env(env_name, policy_kwargs.pop("env_class"), version, color)
    env = get_ray_env(env_name, policy_kwargs.pop("env_class"), version, color)

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=512, callback_after_eval=stop_train_callback, verbose=1)

    if version >= 0:
        policy_kwargs.pop("policy", "MlpPolicy")
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
            tensorboard_log=None  #os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )
    else:
        model = PPO(
            policy_kwargs.pop("policy", "MlpPolicy"),
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
            use_sde=True,
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=None,  # os.path.join(LOG_PATH, f"{env_name}_tensorboard")
        )

    print("optimizing for intel...")
    original_policy = None
    if device == Device.XPU and hasattr(th, "xpu") and th.xpu.is_available():
        ...
#        model.policy, model.policy.optimizer = th.xpu.optimize(
#            model.policy,
#            optimizer=model.policy.optimizer,
#            dtype=th.bfloat16,
#            weights_prepack=False,
#            optimize_lstm=True,
#            # fuse_update_step=True,
#            # auto_kernel_selection=True,
#        )

        # model.policy.optimizer.defaults["momentum"] = SGD_MOMENTUM
        model.policy.optimizer.defaults["weight_decay"] = ADAMW_WEIGHT_DECAY
    else:
        model.policy, model.policy.optimizer = ipex.optimize(
            model.policy,
            optimizer=model.policy.optimizer,
            dtype=th.float16,
            weights_prepack=False,
            optimize_lstm=True,
            # fuse_update_step=True,
            # auto_kernel_selection=True,
        )
        original_policy = model.policy
        model.policy = th.compile(model.policy, backend="ipex")

    # print("optimizing with lightning")
    # fabric = lightning.Fabric(accelerator="cpu", devices=4, strategy="ddp")
    # fabric.launch()
    # model.policy, optimizer = fabric.setup(model.policy, optimizer)

    print("training started...")
    try:
        if device == Device.XPU:
            with th.xpu.amp.autocast(enabled=True, dtype=th.bfloat16):
                model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
        else:
            with th.cpu.amp.autocast(enabled=True, dtype=th.float16):
                model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
    finally:
        if original_policy:
            model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
        model.save(f"./{env_name}.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")

    if hasattr(env, "envs"):
        for e in env.envs:
            e.controller.tear_down()
            e.summarize()


def loop_train(
    env_name: str = "default",
    version: int = -1,
    loops: int = 5,
    color: bool = True,
    device: Device = Device.AUTO.value,
):
    set_start_method("fork")

    policy_kwargs = policy_kwargs_map[env_name]
    policy = policy_kwargs.pop("policy", "MlpPolicy")
    env_class = policy_kwargs.pop("env_class")

    print("loading env...")
    env = get_env(env_name, env_class, version, color)
#    env = get_ray_env(env_name, env_class, version, color)

    for _ in range(loops):
        print("setting up model...")
        t0 = perf_counter()
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

        print("optimizing for intel...")
        original_policy = None
        if device == Device.XPU and hasattr(th, "xpu") and th.xpu.is_available():
            ...
#            model.policy, model.policy.optimizer = th.xpu.optimize(
#                model.policy,
#                optimizer=model.policy.optimizer,
#                dtype=th.bfloat16,
#                weights_prepack=False,
#                optimize_lstm=True,
#                # fuse_update_step=True,
#                # auto_kernel_selection=True,
#            )
        else:
            model.policy, model.policy.optimizer = ipex.optimize(
                model.policy,
                optimizer=model.policy.optimizer,
                dtype=th.float16,
                weights_prepack=False,
                optimize_lstm=True,
                # fuse_update_step=True,
                # auto_kernel_selection=True,
            )
            original_policy = model.policy
            model.policy = th.compile(model.policy, backend="ipex")

        print("training started...")
        try:
            if device == Device.XPU:
                with th.xpu.amp.autocast(enabled=True, dtype=th.bfloat16):
                    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
            else:
                with th.cpu.amp.autocast(enabled=True, dtype=th.float16):
                    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=env_name)  # progress_bar=True
        except:
            print("learning crashed")
            import traceback

            traceback.print_exc()
            # start over with new env
            if hasattr(env, "envs"):
                for e in env.envs:
                    e.controller.tear_down()
            env = get_env(env_name, env_class, version, color)
#            env = get_ray_env(env_name, env_class, version, color)
        else:
            # on success increment the version and keep learning
            version += 1

            # shouldn't be necessary but seems to be
            if hasattr(env, "envs"):
                for e in env.envs:
                    e.controller.tear_down()
            env = get_env(env_name, env_class, version, color)
#            env = get_ray_env(env_name, env_class, version, color)
        finally:
            if original_policy:
                model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
            model.save(f"./{env_name}.v{version + 1}")
            print(f"training finished in: {perf_counter() - t0}")

    if hasattr(env, "envs"):
        for e in env.envs:
            e.controller.tear_down()


def get_env(env_name, env_class, version, color) -> gym.Env:
    print("loading models...")
    models = []
    ext = "Black" if color else "White"
    model_versions = ["A", "B", "C", "D", "E", "F"]
    for model_version in model_versions:
        path = f"Hex9{ext}{model_version}.onnx"
        models.append(ort.InferenceSession(
            path, providers=["OpenVINOExecutionProvider"]
        ) if os.path.exists(path) else None)

    env: Union[DummyVecEnv, SubprocVecEnv] = make_vec_env(
        lambda: env_class(color=color, models=models),
        # monitor_dir=os.path.join(LOG_PATH, env_name, f"v{version}"),
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
        # vec_env_cls=FasterVecEnv,
        # vec_env_kwargs={"executor": executor},
    )
    # env.device = device

    # nice try, but works only with Box and Dict spaces
    # env = VecFrameStack(env, n_stack=8)

    return VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))


def get_ray_env(env_name, env_class, version, color):
    mp.set_start_method('fork')
    env = WrapperRayVecEnv(lambda seed, models=[]: env_class(color=color, models=models), N_ENV_WORKERS, int(N_ENVS // N_ENV_WORKERS), color)
    return VecMonitor(env, os.path.join(LOG_PATH, env_name, f"v{version}"))


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :returns: (numpy array)
    """

    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, "valid")


def plot_results(log_folder, title="Learning Curve"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """

    x, y = ts2xy(load_results(log_folder), "timesteps")
    y = moving_average(y, window=int(N_ENVS * 1.5))  # average across all parallel results
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
        "-c", "--color", type=int, default=1, help="which color player should be trained"
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
        train(args.env, args.version, bool(args.color), args.device)
    elif args.loop_train:
        loop_train(args.env, args.version, args.loops, bool(args.color), args.device)
    elif args.plot:
        path = LOG_PATH if not args.env else os.path.join(LOG_PATH, args.env)
        plot_results(path)
    else:
        print("pick a command")
