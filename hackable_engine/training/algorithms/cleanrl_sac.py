# -*- coding: utf-8 -*-
import os
import random

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.training.hyperparams import *

LOG_PATH = "./hackable_engine/training/logs/"
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod()
            + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = x.flatten(1, -1)
        x = th.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            th.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=th.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            th.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=th.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        x = x.flatten(1, -1)
        mean, log_std = self(x)
        std = log_std.exp()
        normal = th.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = th.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def run(version, policy_kwargs, env, env_name, device, loops, color):
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

    actor = Actor(env).to(device)
    qf1 = SoftQNetwork(env).to(device)
    qf2 = SoftQNetwork(env).to(device)
    qf1_target = SoftQNetwork(env).to(device)
    qf2_target = SoftQNetwork(env).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=Q_LEARNING_RATE
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=LEARNING_RATE)

    env.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        BUFFER_SIZE,
        env.single_observation_space,
        env.single_action_space,
        device,
        n_envs=N_ENVS,
        handle_timeout_termination=False,
    )

    try:
        train(
            env,
            actor,
            actor_optimizer,
            q_optimizer,
            rb,
            writer,
            device,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
        )
    finally:
        th.save(actor.state_dict(), f"./{env_name}.v{version + 1}")

        env.close()
        writer.close()


def train(
    env: gym.vector.VectorEnv,
    actor: Actor,
    actor_optimizer,
    q_optimizer,
    rb: ReplayBuffer,
    writer,
    device,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
):
    # Automatic entropy tuning
    if ENTROPY_AUTOTUNE:
        target_entropy = -th.prod(
            th.Tensor(env.single_action_space.shape).to(device)
        ).item()
        log_alpha = th.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=Q_LEARNING_RATE)
    else:
        alpha = ENTROPY_ALPHA

    obs, _ = env.reset(seed=1)
    for global_step in range(TOTAL_TIMESTEPS):
        # ALGO LOGIC: put action logic here
        if global_step < LEARNING_STARTS:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(N_ENVS)]
            )
        else:
            actions, _, _ = actor.get_action(th.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    break

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs  # for the next iteration

        if global_step > LEARNING_STARTS:
            data = rb.sample(BATCH_SIZE)
            with th.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = (
                    th.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * GAMMA * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % POLICY_FREQUENCY == 0:  # TD 3 Delayed update support
                for _ in range(
                    POLICY_FREQUENCY
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = th.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if ENTROPY_AUTOTUNE:
                        with th.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % TARGET_NETWORK_FREQUENCY == 0:
                for param, target_param in zip(
                    qf1.parameters(), qf1_target.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )
                for param, target_param in zip(
                    qf2.parameters(), qf2_target.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )

            final_rewards = rewards[terminations]
            if final_rewards.size != 0:
                # non_zero_rewards = rewards[np.where(rewards != 0)]
                # print(non_zero_rewards)
                writer.add_scalar("charts/mean_reward", np.mean(final_rewards), global_step)
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                # writer.add_scalar(
                #     "charts/SPS",
                #     int(global_step / (time.time() - start_time)),
                #     global_step,
                # )
                if ENTROPY_AUTOTUNE:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )
