# -*- coding: utf-8 -*-
import asyncio
import os
from argparse import ArgumentParser
from collections import deque

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.training.algorithms.util.replay_buffer import ReplayBuffer
from hackable_engine.training.device import Device
from hackable_engine.training.envs.hex.logit_9_graph_env import Logit9GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import (
    MultiprocessAsyncEnv,
)
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_gat import GraphGAT
from hackable_engine.training.run import LOG_PATH

# th._dynamo.config.cache_size_limit = 16 * 1024 * 1024 * 1024
# th._dynamo.config.suppress_errors = True
# th.set_num_threads(1)

board_size = 9
# gnn_shape = (36, 108, 216, 216)  # for SGConv
gnn_shape = (36, 72, 144, 216)  # for GATConv
# gnn_shape = (32, 64, 128, 256)
mlp_shape = (128,)
num_episodes = 1800
num_envs = 512
num_workers = 4
learning_rate = 2.4e-4
batch_size = 64
gamma_high = 0.95
gamma_low = 0.05
# gamma = 0.95
epsilon = 0.5
epsilon_min = 0.01
target_update_frequency = 128
buffer_size = 5000 * num_envs
start_training = buffer_size * 0.01
epochs = 8

action_queue = deque(maxlen=board_size**2 * num_envs)
# max_action_queue = deque(maxlen=num_envs)
# policy_loss_queue = deque(maxlen=5 * num_envs)
value_loss_queue = deque(maxlen=5 * num_envs)
entropy_queue = deque(maxlen=5 * num_envs)


def get_learning_rate_decay():
    def reverse_sigmoid(episode):
        x = episode / num_episodes
        decay = -0.75 / (1 + np.e ** (-10 * (x - 0.1))) + 1
        return decay

    def warmup_sigmoid(episode):
        warm_up = 0.15
        if episode / num_episodes < warm_up:
            x = episode / (warm_up * num_episodes)
            return 0.66 / (1 + np.e ** (-10 * (x - 0.5))) + 0.34
        x = (episode - warm_up * num_episodes) / ((1 - warm_up) * num_episodes)
        decay = -0.66 / (1 + np.e ** (-8 * (x - 0.4))) + 1
        return decay

    def squared(episode):
        warm_up = 0.2
        if episode / num_episodes < warm_up:
            return 1 - (num_episodes * warm_up - episode) / (num_episodes * warm_up) / 2
        # fmt: off
        return (1 - (episode-warm_up*num_episodes) / ((1-warm_up)*num_episodes) / 3 * 2) ** 2
        # fmt: on

    def one(episode):
        return 1

    return warmup_sigmoid


def epsilon_decay(episode):
    x = episode / num_episodes
    hyperbolic = 1 / (1 + 99 * x**0.75)
    # exponential = 0.995 ** episode
    # return max((hyperbolic, exponential))
    return hyperbolic


# def train():
async def train():
    eps = epsilon
    gamma = gamma_low
    episode_steps = board.size_square
    # if is_logit_env:
    #     episode_steps = board.size_square
    # else:
    #     episode_steps = board.size_square**2 / 2 / epochs
    state, _ = envs.reset()
    state = state.copy()
    # state = state.reshape(num_envs, board.size_square)
    for episode in range(0, num_episodes):
        print("episode", episode, "buffer", replay_buffer.size())
        steps = 0
        while True:
            steps += 1
            q_values = model(th.tensor(state, dtype=th.float32, device=Device.XPU))

            if is_logit_env:
                actions = get_logit_actions(eps, q_values)
            else:
                actions = get_seq_actions(eps, q_values)

            # action_queue.extend(actions.flatten().tolist())
            # state = run_env_sync(state, actions, gamma)
            # loss, gamma, lr = run_train()
            state, (loss, gamma, lr) = await asyncio.gather(
                run_env(state, actions, gamma), run_train()
            )
            if loss is not None and steps > episode_steps:
                value_loss_queue.append(loss.item())

                if is_logit_env:
                    entropy = get_logit_entropy(q_values)
                else:
                    entropy = get_seq_entropy(q_values)
                entropy_queue.append(entropy.item())

                break

        # reduce exploration
        eps = max(epsilon_min, epsilon * epsilon_decay(episode))
        optimizer_scheduler.step()

        if episode % target_update_frequency == 4:
            print("updating target")
            target_model.load_state_dict(model.state_dict())

        if episode % 1 == 0:
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
            gradients = th.cat(gradients)
            gradient = gradients.norm()

            log_progress(episode, episode_steps, gradient, eps, gamma, lr)


async def run_env(states, actions, gamma):
    next_states, rewards, dones, _, _ = await envs.step(actions)
    # next_states = next_states.reshape(num_envs, board.size_square)

    positions = np.random.randint(0, buffer_size, num_envs)
    for position, i in zip(positions, range(num_envs)):
        reward = rewards[i]
        if gamma == gamma_high and reward < -1.0:
            # likely the action was chosen at random, do not add to replay buffer
            continue
        replay_buffer.push(
            position, (states[i], actions[i], reward, next_states[i], dones[i])
        )

    return next_states.copy()


def run_env_sync(states, actions, gamma):
    next_states, rewards, dones, _, _ = envs.step(actions)
    # next_states = next_states.reshape(num_envs, board.size_square)

    positions = np.random.randint(0, buffer_size, num_envs)
    for position, i in zip(positions, range(num_envs)):
        reward = rewards[i]
        if gamma == gamma_high and reward < -1.0:
            # likely the action was chosen at random, do not add to replay buffer
            continue
        replay_buffer.push(
            position, (states[i], actions[i], reward, next_states[i], dones[i])
        )

    return next_states


# def run_train():
async def run_train():
    loss = None
    gamma = gamma_low
    size = replay_buffer.size()
    if size > start_training:
        for i in range(epochs):
            batch = replay_buffer.sample(batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            gamma = calculate_gamma(np.array(rewards), np.array(dones))

            states = th.tensor(np.array(states), dtype=th.float32, device=Device.XPU)
            actions = th.tensor(actions, dtype=th.float32, device=Device.XPU)
            if is_logit_env:
                actions = actions.to(th.int64)
            rewards = th.tensor(rewards, dtype=th.float32, device=Device.XPU)
            next_states = th.tensor(
                np.array(next_states), dtype=th.float32, device=Device.XPU
            )
            dones = th.tensor(dones, dtype=th.float32, device=Device.XPU)

            if is_logit_env:
                loss = get_logit_loss(
                    states, next_states, rewards, dones, actions, gamma
                )
            else:
                loss = get_seq_loss(states, next_states, rewards, dones, gamma)
            # should_raise = False
            # for name, tensor in [
            #     ("states", states),
            #     ("actions", actions),
            #     ("rewards", rewards),
            #     ("dones", dones),
            #     ("next_states", next_states),
            #     ("loss", loss),
            # ]:
            #     if th.any(th.isnan(tensor)):
            #         print(name, tensor)
            #         should_raise = True
            # if should_raise:
            #     raise ValueError("NAN")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        del states, next_states, actions, rewards, dones
    else:
        print(f"{np.round(size / start_training * 100, 2)} %")
    return loss, gamma, optimizer_scheduler.get_last_lr()[0]


def get_logit_actions(eps, q_values):
    return np.array(
        [
            (
                np.random.randint(0, board.size_square)
                if np.random.rand() < eps
                else q_values[i].argmax().item()
            )
            for i in range(num_envs)
        ],
        dtype=np.float32,
    )


def get_seq_actions(eps, q_values):
    actions = to_probs(q_values.to(Device.CPU).detach())
    for _ in range(int(num_envs * eps)):
        actions[np.random.randint(0, num_envs - 1)] = np.random.rand()
    return actions


def get_logit_entropy(q_values):
    probs = th.nn.functional.softmax(q_values.clone().detach(), dim=-1)
    return -th.sum(probs * (probs + 1e-8).log(), dim=-1).mean()


def get_seq_entropy(q_values):
    return th.std(q_values)


def get_logit_loss(states, next_states, rewards, dones, actions, gamma):
    with th.no_grad():
        target_q_values = target_model(next_states)
        target = rewards + gamma * target_q_values.max(dim=1)[0] * (1 - dones)

    q_values = model(states)
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    loss = nn.functional.mse_loss(q_value, target)

    if th.any(th.isnan(loss)):
        print(q_value, target, actions, q_values, states)
        raise ValueError("NAN")
    return loss


def get_seq_loss(states, next_states, rewards, dones, gamma):
    actions = model(states).squeeze(1)
    target_actions = target_model(next_states).squeeze(1).detach()
    target = rewards + gamma * target_actions * (1 - dones)

    return nn.functional.mse_loss(actions, target)


def get_seq_advantage_loss(states, next_states, rewards, dones, gamma):
    actions = model(states).squeeze(1)
    target_actions = target_model(next_states).squeeze(1).detach()
    target = rewards + gamma * target_actions * (1 - dones)
    advantage = target - actions

    loss = -(to_probs(actions) + 1e-8).log() * advantage
    return loss.mean()


def to_probs(actions):
    return nn.functional.sigmoid(actions)
    # mean = actions.mean()
    # std = actions.std()
    # return (nn.functional.tanh((actions - mean)/(std + 1e-8)) + 1) / 2


def calculate_gamma(rewards, dones) -> float:
    if not np.any(dones):
        return gamma_high
    mean_reward = (rewards * dones)[dones == 1].mean().item()
    a = min(max(mean_reward + 1.25, 0), 1)
    gamma = gamma_high * a + gamma_low * (1 - a)
    return gamma


def log_progress(episode, episode_steps, gradient, eps, gamma, lr):
    # episode = episode * episode_steps  # * num_envs
    """steps per env facilitates comparing efficiency of multi-env"""
    # actions = np.array(action_queue)
    # writer.add_histogram(
    #     "actions/raw",
    #     actions,
    #     bins="auto",
    #     max_bins=100,
    # )
    env_progress_data: EnvProgressData = envs.get_progress_data()
    writer.add_scalar("episode/length", env_progress_data.length_mean, episode)
    writer.add_scalar("episode/fps", env_progress_data.time_mean * num_envs, episode)
    # writer.add_scalar("rewards/mean_total", env_progress_data.return_mean, episode)
    writer.add_scalar("rewards/mean_winner", env_progress_data.winner_mean, episode)
    writer.add_scalar("rewards/mean_final", env_progress_data.reward_mean, episode)
    # writer.add_scalar("losses/policy_loss", np.mean(policy_loss_queue), episode)
    writer.add_scalar("losses/value_loss", np.mean(value_loss_queue), episode)
    writer.add_scalar("misc/gamma", gamma, episode)
    writer.add_scalar("misc/epsilon", eps, episode)
    writer.add_scalar("misc/entropy", np.mean(entropy_queue), episode)
    writer.add_scalar("misc/gradient", gradient, episode)
    writer.add_scalar("misc/lr", lr, episode)


def setup_model():
    model = GraphGAT(
        node_count=board.size_square,
        node_features=9,
        output_size=1,
        batch_size=batch_size,
        num_envs=num_envs,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
        edge_types=nn.functional.one_hot(board.edge_types, num_classes=3),
    ).to(Device.XPU)
    target_model = GraphGAT(
        node_count=board.size_square,
        node_features=9,
        output_size=1,
        batch_size=batch_size,
        num_envs=num_envs,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
        edge_types=nn.functional.one_hot(board.edge_types, num_classes=3),
    ).to(Device.XPU)

    # model = th.compile(model)#, mode="max-autotune")
    # target_model = th.compile(target_model)#, mode="max-autotune")
    # model = GraphGAT(
    #     input_size=3,
    #     output_size=1,
    #     batch_size=batch_size,
    #     num_envs=num_envs,
    #     graph_shape=(18, 72, 144, 216),
    #     mlp_shape=(128, 128),
    #     edge_index=board.edge_index,
    #     edge_types=board.edge_types,
    # ).to(Device.XPU)
    # target_model = GraphGAT(
    #     input_size=3,
    #     output_size=1,
    #     batch_size=batch_size,
    #     num_envs=num_envs,
    #     graph_shape=(18, 72, 144, 216),
    #     mlp_shape=(128, 128),
    #     edge_index=board.edge_index,
    #     edge_types=board.edge_types,
    # ).to(Device.XPU)

    # optimizer = optim.SGD(
    #     model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
    # )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(
        capacity=buffer_size, storage_type=ReplayBuffer.StorageType.LIST
    )
    return model, target_model, optimizer, replay_buffer


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--env",
        help="monitor logs subpath to use for the plot",
        type=str,
        default="",
    )
    parser.add_argument(
        "-l", "--load-version", type=int, default=0, help="version of the model to load"
    )
    parser.add_argument(
        "-v", "--version", type=int, default=0, help="version of the model to save"
    )
    parser.add_argument(
        "-c",
        "--color",
        type=int,
        default=1,
        help="which color player should be trained",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    board = HexBoard("", size=board_size, use_graph=True)
    writer = SummaryWriter(
        # os.path.join(LOG_PATH, f"dqn_tensorboard_{board_size}", f"dqn_v{args.version}")
        os.path.join(LOG_PATH, f"dqn_tensorboard_{board_size}", f"dqn_v{args.version}")
    )

    is_logit_env = True
    envs = MultiprocessAsyncEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            SyncVectorEnv(
                [
                    lambda: Logit9GraphEnv(color=False, models=[None])
                    for _ in range(num_envs)
                ],
                copy=False,
            ),
            is_multiprocessed=True,
        ),
        num_workers,
        int(num_envs // num_workers),
        action_shape=(1,),
        color=False,
    )
    # envs = EpisodeStats(
    #     SyncVectorEnv(
    #         [lambda: Logit9GraphEnv(color=False, models=[None]) for _ in range(num_envs)],
    #         copy=False,
    #     ),
    #     is_multiprocessed=False,
    # )
    model, target_model, optimizer, replay_buffer = setup_model()
    optimizer_scheduler = LambdaLR(optimizer, get_learning_rate_decay())
    try:
        # with th.amp.autocast(Device.XPU.value, enabled=True, dtype=th.bfloat16):
            # train()
        asyncio.run(train())
    finally:
        writer.close()

        if not os.path.exists("simple_dqn"):
            os.mkdir("simple_dqn")
        color = "white" if args.color else "black"
        th.save(model.state_dict(), f"simple_dqn/dqn-model-{color}.v{args.version}")
        th.save(
            target_model.state_dict(),
            f"simple_dqn/dqn-target-model-{color}.v{args.version}",
        )
        th.save(
            optimizer.state_dict(), f"simple_dqn/dqn-optimizer-{color}.v{args.version}"
        )
