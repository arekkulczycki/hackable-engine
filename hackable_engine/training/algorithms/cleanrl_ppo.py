# -*- coding: utf-8 -*-
import os
import random
from contextlib import nullcontext

import numpy as np
import torch as th
from gymnasium.vector import VectorEnv
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.training.device import Device
from hackable_engine.training.hyperparams import *

LOG_PATH = "./hackable_engine/training/logs/"


class Agent(th.nn.Module):
    def __init__(
        self,
        envs: VectorEnv,
        mlp_net_arch: tuple[int, int],
        features_extractor_class,
        features_extractor_kwargs,
    ):
        super().__init__()
        self.critic = th.nn.Sequential(
            self.layer_init(
                th.nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    mlp_net_arch[0],
                )
            ),
            th.nn.Tanh(),
            self.layer_init(th.nn.Linear(mlp_net_arch[0], mlp_net_arch[1])),
            th.nn.Tanh(),
            self.layer_init(th.nn.Linear(mlp_net_arch[1], 1), std=1.0),
        )
        self.actor_mean = th.nn.Sequential(
            self.layer_init(
                th.nn.Linear(
                    np.array(envs.single_observation_space.shape).prod(),
                    mlp_net_arch[0],
                )
            ),
            th.nn.Tanh(),
            self.layer_init(th.nn.Linear(mlp_net_arch[0], mlp_net_arch[1])),
            th.nn.Tanh(),
            self.layer_init(
                th.nn.Linear(mlp_net_arch[1], np.prod(envs.single_action_space.shape)),
                std=0.01,
            ),
        )
        self.actor_logstd = th.nn.Parameter(
            th.zeros(1, np.prod(envs.single_action_space.shape))
        )

    @staticmethod
    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        th.nn.init.orthogonal_(layer.weight, std)
        th.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        x = x.flatten(1, -1)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.flatten(1, -1)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = th.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


def run(version, policy_kwargs, env, env_name, device, loops, color):
    base_path, new_path = init_directory(env_name, version)
    writer = SummaryWriter(
        os.path.join(LOG_PATH, f"{env_name}_tensorboard", f"{env_name}_{version}")
    )
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    random.seed(1)
    np.random.seed(1)
    th.manual_seed(1)

    agent = Agent(
        env,
        mlp_net_arch=policy_kwargs["net_arch"],
        features_extractor_class=policy_kwargs.get("features_extractor_class"),
        features_extractor_kwargs=policy_kwargs.get("features_extractor_kwargs"),
    ).to(device)

    optimizer = Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = th.zeros((N_STEPS, N_ENVS) + env.single_observation_space.shape).to(device)
    actions = th.zeros((N_STEPS, N_ENVS) + env.single_action_space.shape).to(device)
    logprobs = th.zeros((N_STEPS, N_ENVS)).to(device)
    rewards = th.zeros((N_STEPS, N_ENVS)).to(device)
    dones = th.zeros((N_STEPS, N_ENVS)).to(device)
    values = th.zeros((N_STEPS, N_ENVS)).to(device)

    load_model_if_available(base_path, env_name, version, agent, optimizer)

    try:
        context = (
            th.amp.autocast("xpu", enabled=True, dtype=th.float16)
            if device == Device.XPU
            else nullcontext()
        )
        with context:
            train(
                env,
                agent,
                optimizer,
                writer,
                device,
                obs,
                actions,
                values,
                rewards,
                dones,
                logprobs,
            )
    finally:
        save_models(
            new_path,
            env_name,
            version,
            agent,
            optimizer,
        )

        env.close()
        writer.close()


def load_model_if_available(base_path, env_name, version, agent, optimizer):
    if os.path.exists(base_path):
        print("loading pre-trained weights...")
        agent.load_state_dict(
            th.load(f"{base_path}/{env_name}-ppo-agent.v{version}", weights_only=True)
        )
        optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-ppo-optimizer.v{version}", weights_only=True
            )
        )


def save_models(new_path, env_name, version, agent, optimizer):
    th.save(agent.state_dict(), f"{new_path}/{env_name}-ppo-agent.v{version + 1}")
    th.save(
        optimizer.state_dict(), f"{new_path}/{env_name}-ppo-optimizer.v{version + 1}"
    )


def init_directory(env_name, version) -> tuple[str, str]:
    base_path = f"./{env_name}-sac.v{version}"
    new_path = f"./{env_name}-sac.v{version + 1}"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        try:
            with open(f"{new_path}/hyperparams.log", "w") as f:
                for param in [
                    N_ENVS,
                    TOTAL_TIMESTEPS,
                    LEARNING_RATE,
                    MAX_GRAD_NORM,
                    N_STEPS,
                    BUFFER_SIZE,
                    LEARNING_STARTS,
                    BATCH_SIZE,
                    GAMMA,
                    CLIP_RANGE,
                    GAE_LAMBDA,
                    ENT_COEF,
                ]:
                    name = get_variable_name(param)
                    if name:
                        f.write(f"{name}={param}\n")
        except ValueError:
            print("hyperparams file not created")
    return base_path, new_path


def get_variable_name(v):
    for pair in locals().copy():
        try:
            name, val = pair
            if v is val:
                return name
        except ValueError:
            continue
    raise ValueError("variable not found")


def train(
    env: VectorEnv,
    agent: Agent,
    optimizer,
    writer,
    device,
    obs,
    actions,
    values,
    rewards,
    dones,
    logprobs,
):
    global_step = 0
    next_obs, _ = env.reset(seed=1)
    next_obs = th.tensor(next_obs).to(device)
    next_done = th.zeros(N_ENVS).to(device)
    num_iterations = TOTAL_TIMESTEPS // N_STEPS

    for iteration in range(1, num_iterations + 1):
        # # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / args.num_iterations
        #     lrnow = frac * LEARNING_RATE
        #     optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, N_STEPS):
            global_step += N_ENVS
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with th.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = env.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = th.tensor(reward).to(device).view(-1)
            next_obs, next_done = th.Tensor(next_obs).to(device), th.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        # bootstrap value if not done
        with th.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = th.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(N_STEPS)):
                if t == N_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(N_STEPS * N_ENVS)
        clipfracs = []
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, N_STEPS * N_ENVS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if NORMALIZE_ADVANTAGES:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(
                    ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE
                )
                pg_loss = th.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if CLIP_VLOSS:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + th.clamp(
                        newvalue - b_values[mb_inds],
                        -CLIP_RANGE,
                        CLIP_RANGE,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            # if args.target_kl is not None and approx_kl > args.target_kl:
            #     break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        all_env_steps = global_step
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_histogram(
            "charts/actions",
            np.array(env.action_queue),
            bins="auto",
            max_bins=100,
        )
        writer.add_scalar(
            "charts/episode_len", np.mean(env.length_queue), all_env_steps
        )
        writer.add_scalar(
            "charts/episode_fps", np.mean(env.time_queue) * N_ENVS, all_env_steps
        )
        writer.add_scalar(
            "charts/mean_return", np.mean(env.return_queue), all_env_steps
        )
        writer.add_scalar(
            "charts/mean_winner", np.mean(env.winner_queue), all_env_steps
        )
        writer.add_scalar(
            "charts/mean_reward", np.mean(env.reward_queue), all_env_steps
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), all_env_steps)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), all_env_steps)
        writer.add_scalar("losses/entropy", entropy_loss.item(), all_env_steps)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), all_env_steps)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), all_env_steps)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), all_env_steps)
        writer.add_scalar("losses/explained_variance", explained_var, all_env_steps)

        # print(f"mean_reward={mean_reward}, explained_var={explained_var}, approx_kl={approx_kl}, value_loss={v_loss.item()}, policy_loss={pg_loss.item()}, entropy_loss={entropy_loss.item()}")
