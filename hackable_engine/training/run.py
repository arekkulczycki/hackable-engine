# -*- coding: utf-8 -*-
import multiprocessing as mp
import os
from argparse import ArgumentParser
from enum import Enum
from time import perf_counter
from typing import Union

import gym
import intel_extension_for_pytorch as ipex
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core
import torch as th
from RayEnvWrapper import WrapperRayVecEnv
from stable_baselines3 import PPO

# from sbx import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import ts2xy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
)
from torch.optim import AdamW, SGD

from hackable_engine.training.callbacks import TensorboardActionHistogramCallback
from hackable_engine.training.hyperparams import *
from hackable_engine.training.policies import policy_kwargs_map

# from stable_baselines3.common.callbacks import (
#     EvalCallback,
#     StopTrainingOnNoModelImprovement,
# )

LOG_PATH = "./hackable_engine/training/logs/"
SEARCH_LIMIT = 9
"""In the case of using tree search in training."""


class Device(str, Enum):
    GPU = "cuda"
    CPU = "cpu"
    AUTO = "auto"
    XPU = "xpu"  # intel Arc GPU


def train(
    env_name: str = "default",
    version: int = -1,
    color: bool = True,
    device: Device = Device.AUTO.value,
    loops: int = 100,
):
    # mp.set_start_method("fork")
    t0 = perf_counter()

    policy_kwargs = policy_kwargs_map[env_name]

    print("loading env...")
    if N_ENV_WORKERS == 1:
        env = get_env(env_name, policy_kwargs.pop("env_class"), version, color)
    else:
        env = get_ray_env(env_name, policy_kwargs.pop("env_class"), version, color)

    # Stop training if there is no improvement after more than 3 evaluations
    # stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    # eval_callback = EvalCallback(env, eval_freq=512, callback_after_eval=stop_train_callback, verbose=1)

    print("configuring ppo...")
    if version >= 0:
        reset_num_timesteps = RESET_CHARTS
        policy_class = policy_kwargs.pop("policy", "MlpPolicy")
        policy_kwargs["should_initialize_weights"] = False
        if policy_class == "CnnPolicy":
            policy_kwargs["features_extractor_kwargs"][
                "should_initialize_weights"
            ] = False
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
                "max_grad_norm": MAX_GRAD_NORM,
            },
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )
    else:
        reset_num_timesteps = True
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
            max_grad_norm=MAX_GRAD_NORM,
            # use_sde=True,
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )

    if model.policy.optimizer.__class__ is SGD:
        model.policy.optimizer.defaults["momentum"] = SGD_MOMENTUM[1]
        model.policy.optimizer.defaults["dampening"] = SGD_DAMPENING[1]
    if model.policy.optimizer.__class__ is AdamW:
        model.policy.optimizer.defaults["weight_decay"] = ADAMW_WEIGHT_DECAY[1]

    print("running initial train loop...")
    model.policy.train()

    print("optimizing for intel...")
    original_policy = None
    if device == Device.XPU and hasattr(th, "xpu") and th.xpu.is_available():
        # this actually seems necessary to have a non-None gradient in the conv layer
        model.policy, model.policy.optimizer = th.xpu.optimize(
            model.policy,
            optimizer=model.policy.optimizer,
            # dtype=th.bfloat16,
            # weights_prepack=False,
            # optimize_lstm=True,
            # fuse_update_step=True,
            # auto_kernel_selection=True,
            # split_master_weight_for_bf16=True,
        )
        # if model.policy.share_features_extractor:
        #     model.policy.features_extractor = th.compile(model.policy.features_extractor, backend="ipex")
        # original_policy = model.policy
        # model.policy = th.compile(model.policy, backend="ipex")
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
            with th.xpu.amp.autocast(enabled=True, dtype=th.float16):
            # with th.xpu.amp.autocast(enabled=True, dtype=th.bfloat16):
                for _ in range(loops):
                    model.learn(
                        total_timesteps=TOTAL_TIMESTEPS,
                        reset_num_timesteps=reset_num_timesteps,
                        tb_log_name=env_name,
                        callback=TensorboardActionHistogramCallback(
                            color, should_log_actions=True
                        ),
                    )  # progress_bar=True
                    reset_num_timesteps = False
                    if original_policy:
                        model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
                    model.save(f"./{env_name}.v{version + 1}.checkpoint")
        else:
            with th.cpu.amp.autocast(enabled=True, dtype=th.float16):
                for _ in range(loops):
                    model.learn(
                        total_timesteps=TOTAL_TIMESTEPS,
                        reset_num_timesteps=reset_num_timesteps,
                        tb_log_name=env_name,
                        callback=TensorboardActionHistogramCallback(color),
                    )  # progress_bar=True
                    reset_num_timesteps = False
                    if original_policy:
                        model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
                    model.save(f"./{env_name}.v{version + 1}.checkpoint")
    finally:
        model.save(f"./{env_name}.v{version + 1}")

    print(f"training finished in: {perf_counter() - t0}")

    if hasattr(env, "envs"):
        for e in env.envs:
            e.controller.tear_down()
            e.summarize()


def get_env(env_name, env_class, version, color) -> gym.Env:
    print("loading models...")
    # ort_ses_opt = ort.SessionOptions()
    # ort_ses_opt.log_severity_level = 3
    # ort_ses_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    ie = Core()

    models = []
    color_ext = "Black" if color else "White"
    for model_version in ["A", "B", "C", "D", "E", "F"]:
        path = f"Hex9{color_ext}{model_version}.onnx"
        # models.append(
        #     ort.InferenceSession(
        #         path, providers=["CPUExecutionProvider"], sess_options=ort_ses_opt
        #     )
        # )
        model = ie.compile_model(model=ie.read_model(model=path), device_name="CPU")  # GPU
        models.append(model)

    env: Union[DummyVecEnv, SubprocVecEnv] = make_vec_env(
        # lambda: env_class(color=color, models=models),
        env_class,
        env_kwargs=dict(color=color, models=models),
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
    env = WrapperRayVecEnv(
        lambda seed, models=[]: env_class(color=color, models=models),
        N_ENV_WORKERS,
        int(N_ENVS // N_ENV_WORKERS),
        color,
    )
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
    y = moving_average(
        y, window=int(N_ENVS * 1.5)
    )  # average across all parallel results
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
        "-l", "--loops", type=int, default=100, help="iteration of training to perform"
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
    elif args.plot:
        path = LOG_PATH if not args.env else os.path.join(LOG_PATH, args.env)
        plot_results(path)
    else:
        print("pick a command")
