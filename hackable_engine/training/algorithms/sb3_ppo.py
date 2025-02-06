# -*- coding: utf-8 -*-
import os

import torch as th
from stable_baselines3 import PPO
from torch.optim import AdamW, SGD

from hackable_engine.training.device import Device
from hackable_engine.training.callbacks import TensorboardActionHistogramCallback
from hackable_engine.training.hyperparams import *
from hackable_engine.training.policies import policy_kwargs_map

LOG_PATH = "./hackable_engine/training/logs/"


def run(version, policy_kwargs, env, env_name, device, loops, color):

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

    for module in model.policy.mlp_extractor.policy_net:
        if isinstance(module, th.nn.Linear):
            # th.nn.init.xavier_normal_(module.weight, gain=1.0)
            th.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )

            if module.bias is not None:
                th.nn.init.zeros_(module.bias)
    for module in model.policy.mlp_extractor.value_net:
        if isinstance(module, th.nn.Linear):
            # th.nn.init.xavier_normal_(module.weight, gain=1.0)
            th.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )

            if module.bias is not None:
                th.nn.init.zeros_(module.bias)

    print("running initial train loop...")
    model.policy.train()

    # print("optimizing for intel...")
    # original_policy = None
    # if device == Device.XPU and hasattr(th, "xpu") and th.xpu.is_available():
    #     # this actually seems necessary to have a non-None gradient in the conv layer
    #     model.policy, model.policy.optimizer = th.xpu.optimize(
    #         model.policy,
    #         optimizer=model.policy.optimizer,
    #         # dtype=th.bfloat16,
    #         # weights_prepack=False,
    #         # optimize_lstm=True,
    #         # fuse_update_step=True,
    #         # auto_kernel_selection=True,
    #         # split_master_weight_for_bf16=True,
    #     )
    #     # if model.policy.share_features_extractor:
    #     #     model.policy.features_extractor = th.compile(model.policy.features_extractor, backend="ipex")
    #     # original_policy = model.policy
    #     # model.policy = th.compile(model.policy, backend="ipex")
    # else:
    #     model.policy, model.policy.optimizer = ipex.optimize(
    #         model.policy,
    #         optimizer=model.policy.optimizer,
    #         dtype=th.float16,
    #         weights_prepack=False,
    #         optimize_lstm=True,
    #         # fuse_update_step=True,
    #         # auto_kernel_selection=True,
    #     )
    #     original_policy = model.policy
    #     model.policy = th.compile(model.policy, backend="ipex")

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