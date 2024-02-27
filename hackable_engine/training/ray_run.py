# -*- coding: utf-8 -*-

from ray import air, tune
from ray.rllib import VectorEnv
from ray.rllib.algorithms import ppo, PPOConfig

from hackable_engine.training.envs.hex.raw_7x7_bin_gymnasium import Raw7x7BinGymnasium


env = VectorEnv(Raw7x7BinGymnasium.observation_space, Raw7x7BinGymnasium.action_space, 8)


# config = PPOConfig()
# config.lr = 1e-3
# config.num_envs_per_worker = 8
# algo = ppo.PPO(env=Raw7x7BinGymnasium, config=config)
# algo.train()

tune.registry.register_env("raw7x7hex", lambda config: Raw7x7BinGymnasium())
analysis = tune.Tuner(
    "PPO",
    run_config=air.RunConfig(
        stop={"timesteps_total": 2**19},
        local_dir="/home/arek/old/arek-chess/ray_trained_models",
        checkpoint_config=air.CheckpointConfig(
            checkpoint_at_end=True,
        ),
        verbose=3,
    ),
    param_space={"env": "raw7x7hex", "lr": 1e-3},
).fit()
# retrieve the checkpoint path
analysis.default_metric = "episode_reward_mean"
analysis.default_mode = "max"
checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
print(f"Trained model saved at {checkpoint_path}")
