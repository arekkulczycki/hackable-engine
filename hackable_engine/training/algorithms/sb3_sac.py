# -*- coding: utf-8 -*-
import os

import torch as th
from stable_baselines3 import SAC

from hackable_engine.training.device import Device
from hackable_engine.training.callbacks import TensorboardActionHistogramCallback
from hackable_engine.training.hyperparams import *
from hackable_engine.training.policies import policy_kwargs_map

LOG_PATH = "./hackable_engine/training/logs/"


def run(version, policy_kwargs, env, env_name, device, loops, color):

    print("configuring sac...")
    if version >= 0:
        reset_num_timesteps = RESET_CHARTS
        policy_class = policy_kwargs.pop("policy", "MlpPolicy")
        policy_kwargs["should_initialize_weights"] = False
        if policy_class == "CnnPolicy":
            policy_kwargs["features_extractor_kwargs"][
                "should_initialize_weights"
            ] = False
        model = SAC.load(
            f"./{env_name}.v{version}",
            env=env,
            verbose=2,
            param_override={
                "learning_rate": LEARNING_RATE,
                "batch_size": BATCH_SIZE,
                "gamma": GAMMA,
                "ent_coef": ENT_COEF,
            },
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )
    else:
        reset_num_timesteps = True
        model = SAC(
            policy_kwargs.pop("policy", "MlpPolicy"),
            env=env,
            verbose=2,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            gamma=GAMMA,
            ent_coef=ENT_COEF,
            # use_sde=True,
            policy_kwargs=policy_kwargs_map[env_name],
            device=device,
            tensorboard_log=os.path.join(LOG_PATH, f"{env_name}_tensorboard"),
        )

    print("running initial train loop...")
    model.policy.train()

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
                    # if original_policy:
                    #     model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
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
                    # if original_policy:
                    #     model.policy = original_policy  # the state_dict is maintained, therefore saving the trained network
                    model.save(f"./{env_name}.v{version + 1}.checkpoint")
    finally:
        model.save(f"./{env_name}.v{version + 1}")