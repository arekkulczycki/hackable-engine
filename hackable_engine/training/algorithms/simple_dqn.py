# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gymnasium as gym
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.training.device import Device
from hackable_engine.training.envs.hex.logit_7_graph_env import Logit7GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_env import (
    MultiprocessEnv,
)
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.mlp import MLP
from hackable_engine.training.run import LOG_PATH

num_episodes = 20000
num_envs = 4
num_workers = 4
learning_rate = 1.5e-4
batch_size = 64
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9975
target_update_frequency = 500
buffer_size = 10000

action_queue = deque(maxlen=25 * num_envs)
max_action_queue = deque(maxlen=num_envs)
policy_loss_queue = deque(maxlen=5 * num_envs)
value_loss_queue = deque(maxlen=5 * num_envs)
entropy_queue = deque(maxlen=5 * num_envs)


def get_learning_rate_decay():
    def wrapped(episode):
        # return 1
        # return max((1 - episode / num_episodes) ** 2, 0.01)
        warm_up = 0.05
        if episode / num_episodes < warm_up:
            return 1 - (num_episodes * warm_up - episode) / (num_episodes * warm_up) / 3
        return (1 - episode / ((1 - warm_up) * num_episodes) * 2 / 3) ** 2

    return wrapped


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)


def train():
    eps = epsilon
    episode_steps = board.size_square
    for episode in range(num_episodes):
        state, _ = envs.reset()
        state = state.reshape(num_envs, board.size_square)

        steps = 0
        while True:
            steps += 1
            q_values = model(th.tensor(state, dtype=th.float32, device=Device.XPU))
            action_queue.extend(q_values.flatten().tolist())

            actions = np.array(
                [
                    (
                        np.random.randint(0, board.size_square)
                        if np.random.rand() < eps
                        else q_values[i].argmax().item()
                    )
                    for i in range(len(state))
                ]
            )

            next_state, reward, done, truncated, info = envs.step(actions)
            next_state = next_state.reshape(num_envs, board.size_square)

            for i in range(len(state)):
                replay_buffer.push(
                    (state[i], actions[i], reward[i], next_state[i], done[i])
                )

            state = next_state

            if replay_buffer.size() > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = th.tensor(
                    np.array(states), dtype=th.float32, device=Device.XPU
                )
                actions = th.tensor(actions, dtype=th.float32, device=Device.XPU).to(
                    th.int64
                )
                rewards = th.tensor(rewards, dtype=th.float32, device=Device.XPU)
                next_states = th.tensor(
                    np.array(next_states), dtype=th.float32, device=Device.XPU
                )
                dones = th.tensor(dones, dtype=th.float32, device=Device.XPU)

                with th.no_grad():
                    target_q_values = target_model(next_states)
                    target = rewards + gamma * target_q_values.max(dim=1)[0] * (
                        1 - dones
                    )

                q_values = model(states)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = nn.MSELoss()(q_value, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_scheduler.step()

            if steps > episode_steps:
                value_loss_queue.append(loss.item())

                probs = th.nn.functional.softmax(q_values.clone().detach())
                entropy = -th.sum(probs * (probs + 1e-8).log(), dim=-1).mean()
                entropy_queue.append(entropy.item())

                break

        # reduce exploration
        eps = max(epsilon_min, eps * epsilon_decay)

        if episode % target_update_frequency == 0:
            target_model.load_state_dict(model.state_dict())

        if episode % 10 == 0:
            gradients = []
            for param in model.parameters():
                if param.grad is not None:
                    gradients.append(param.grad.view(-1))
            gradients = th.cat(gradients)
            gradient = gradients.norm()

            log_progress(episode, episode_steps, gradient, eps)


def log_progress(episode, episode_steps, gradient, eps):
    steps = episode * episode_steps
    actions = np.array(action_queue)
    writer.add_histogram(
        "actions/probs",
        actions,
        bins="auto",
        max_bins=100,
    )
    writer.add_scalar("episode/length", np.mean(envs.length_queue), steps)
    writer.add_scalar("episode/fps", np.mean(envs.time_queue) * num_envs, steps)
    # writer.add_scalar("rewards/mean_total", np.mean(env.return_queue), all_env_steps)
    writer.add_scalar("rewards/mean_winner", np.mean(envs.winner_queue), steps)
    writer.add_scalar("rewards/mean_final", np.mean(envs.reward_queue), steps)
    # writer.add_scalar("losses/policy_loss", np.mean(policy_loss_queue), steps)
    writer.add_scalar("losses/value_loss", np.mean(value_loss_queue), steps)
    writer.add_scalar("misc/epsilon", eps, steps)
    writer.add_scalar("misc/entropy", np.mean(entropy_queue), steps)
    writer.add_scalar("misc/gradient", gradient, steps)
    writer.add_scalar("misc/lr", get_learning_rate_decay()(episode), steps)


def setup_model():
    model = MLP(
        input_size=board.size_square, output_size=board.size_square, hidden_size=512
    ).to(Device.XPU)

    target_model = MLP(
        input_size=board.size_square, output_size=board.size_square, hidden_size=512
    ).to(Device.XPU)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(capacity=buffer_size)
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

    board_size = 7
    board = HexBoard("", size=board_size, use_graph=True)
    writer = SummaryWriter(
        os.path.join(LOG_PATH, f"dqn_tensorboard", f"dqn_v{args.version}")
    )

    envs = EpisodeStats(
        gym.make_vec(
            id=Logit7GraphEnv.__name__,
            num_envs=num_envs,
            color=False,
            models=[None],
            vectorization_mode=gym.VectorizeMode.SYNC,
        )
    )
    # envs = MultiprocessEnv(
    #     lambda seed, num_envs, color_, models=[]: EpisodeStats(
    #         gym.make_vec(
    #             id=Logit5GraphEnv.__name__,
    #             num_envs=num_envs,
    #             color=color_,
    #             models=models,
    #         ),
    #         is_multiprocessed=True,
    #     ),
    #     num_workers,
    #     int(num_envs // num_workers),
    #     False,
    # )
    model, target_model, optimizer, replay_buffer = setup_model()
    optimizer_scheduler = LambdaLR(optimizer, get_learning_rate_decay())
    try:
        train()
    finally:
        writer.close()

        if not os.path.exists("simple_dqn"):
            os.mkdir("simple_dqn")
        th.save(model.state_dict(), f"simple_dqn/dqn-model-black.v{args.version}")
        th.save(
            target_model.state_dict(),
            f"simple_dqn/dqn-target-model-black.v{args.version}",
        )
        th.save(
            optimizer.state_dict(), f"simple_dqn/dqn-optimizer-black.v{args.version}"
        )
