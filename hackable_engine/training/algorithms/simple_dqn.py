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
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.algorithms.util.replay_buffer import ReplayBuffer
from hackable_engine.training.constants import LRShape, get_learning_rate_decay
from hackable_engine.training.device import Device
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.envs.hex.logit_9_graph_env import Logit9GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import (
    MultiprocessAsyncEnv,
)
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_gat import GraphGAT
from hackable_engine.training.models.graph_gin import GraphGIN
from hackable_engine.training.models.graph_gine import GraphGINE
from hackable_engine.training.models.graph_rgcn import GraphRGCN
from hackable_engine.training.models.graph_sg import GraphSG
from hackable_engine.training.run import LOG_PATH

# th._dynamo.config.cache_size_limit = 16 * 1024 * 1024 * 1024
# th._dynamo.config.suppress_errors = True
# th.set_num_threads(1)

cnn_shape = (256, 512, 1024)
cnn_kernels = (5, 3, 3)
cnn_strides = (1, 1, 1, 1)
cnn_paddings = (0, 0, 0, 0)
GNN_SHAPES = {
    GraphSG: (54, 108, 216, 432, 432),  # 54 for 9 features, 36 for 3
    GraphRGCN: (54, 108, 216, 216),
    GraphGAT: (36, 72, 144, 216),
    GraphGIN: (108, 108, 108, 108),
    GraphGINE: (108, 108, 108, 108),
}

board_size = 11
model_class = GraphSG
gnn_shape = GNN_SHAPES[model_class]
mlp_shape = (128,) # board_size ** 2)
num_episodes = 1024
episode_env_steps = board_size ** 2
base_num_envs = 1024  # GraphSG runs on 1024, GraphGAT on 512, CNN on 2048, to be comparable
"""relative for tensorboard graphs, such that training sessions are comparable"""
num_envs = 1024
num_workers = 4
learning_rate = 2e-4
lr_shape = LRShape.WARMUP_SIGMOID
batch_size = 64
gamma_low = 0.05
gamma_high = 0.95
epsilon = 0.5
epsilon_min = 0.005
target_update_frequency = 128
buffer_size_low = 30 * episode_env_steps * base_num_envs
buffer_size_high = 30 * episode_env_steps * base_num_envs
start_training = buffer_size_low * 0.05
epochs = 8
episode_step = num_envs // base_num_envs
assert episode_step >= 1

# action_queue = deque(maxlen=board_size**2 * num_envs)
# max_action_queue = deque(maxlen=num_envs)
# policy_loss_queue = deque(maxlen=5 * num_envs)
value_loss_queue = deque(maxlen=5 * num_envs)
entropy_queue = deque(maxlen=5 * num_envs)


def epsilon_decay(episode):
    x = episode / num_episodes
    hyperbolic = 10 / (1 + 999 * x ** 0.5)
    # exponential = 0.995 ** episode
    # return max((hyperbolic, exponential))
    return min(1, hyperbolic)


class DQN:
    def __init__(self, envs):
        self.envs = envs
        self.model = model_class(  # does best with high learning rates like 6e-4
            node_count=board.size_square,
            node_features=9,
            output_size=1,
            batch_size=batch_size,
            num_envs=num_envs,
            gnn_shape=gnn_shape,
            # gnn_heads=6,
            mlp_shape=mlp_shape,
            edge_index=board.edge_index,
            # edge_types=board.edge_types,
            # use_res=True,
        ).to(Device.XPU).to(th.float32)
        self.target_model = model_class(
            node_count=board.size_square,
            node_features=9,
            output_size=1,
            batch_size=batch_size,
            num_envs=num_envs,
            gnn_shape=gnn_shape,
            # gnn_heads=6,
            mlp_shape=mlp_shape,
            edge_index=board.edge_index,
            # edge_types=board.edge_types,
            # use_res=True,
        ).to(Device.XPU).to(th.float32)

        th._dynamo.reset()
        self.model = th.compile(self.model)
        self.target_model = th.compile(self.target_model)

        # optimizer = optim.SGD(
        #     model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
        # )
        # TODO: use weight_decay for models which tend to have growing gradient towards the end
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size_low, storage_type=ReplayBuffer.StorageType.LIST
        )
        self.optimizer_scheduler = LambdaLR(self.optimizer, get_learning_rate_decay(lr_shape, num_episodes))
        self.obs: list[np.ndarray] = []

    def run(self):
        asyncio.run(self.train())

    async def train(self):
        eps = epsilon
        gamma = gamma_low
        self.obs = self.envs.reset()

        for episode in range(0, num_episodes, episode_step):
            print("episode", episode, "buffer", self.replay_buffer.size())
            steps = 0
            while True:
                steps += 1
                q_values = self.model(
                    th.from_numpy(np.stack(self.obs, 0)).to(Device.XPU)
                )

                if is_logit_env:
                    actions = self.get_logit_actions(eps, q_values)
                else:
                    actions = self.get_seq_actions(eps, q_values)
                # action_queue.extend(actions.flatten().tolist())

                # obs = self.run_env_sync(self.obs, actions, gamma)
                # loss, gamma, lr = await self.train_on_buffer()
                self.obs, (loss, gamma, lr) = await asyncio.gather(
                    self.run_env(self.obs, actions, gamma), self.train_on_buffer()
                )

                if loss is not None and steps > episode_env_steps:
                    value_loss_queue.append(loss.item())

                    if is_logit_env:
                        entropy = self.get_logit_entropy(q_values)
                    else:
                        entropy = self.get_seq_entropy(q_values)
                    entropy_queue.append(entropy.item())

                    break

            eps = max(epsilon_min, epsilon * epsilon_decay(episode))
            self.optimizer_scheduler.step()

            if episode % target_update_frequency == 4:
                print("updating target")
                self.target_model.load_state_dict(self.model.state_dict())
                if episode > target_update_frequency:
                    self.replay_buffer.capacity = buffer_size_high

            if episode % 1 == 0:
                gradients = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        gradients.append(param.grad.view(-1))
                gradients = th.cat(gradients)
                gradient = gradients.norm()

                log_progress(episode, episode_env_steps, gradient, eps, gamma, lr)

    async def run_env(self, last_obss, actions, gamma):
        obss, rewards, dones, _, _ = await envs.step(actions)

        positions = np.random.randint(0, self.replay_buffer.capacity, num_envs)
        for position, last_obs, action, reward, obs, done in zip(
            positions, last_obss, actions, rewards, obss, dones
        ):
            if gamma == gamma_high and reward < -1.0:
                # likely the action was chosen at random, do not add to replay buffer
                continue
            self.replay_buffer.push(position, (last_obs, action, reward, obs, done))

        return obss

    def run_env_sync(self, last_obss, actions, gamma):
        obss, rewards, dones, _, _ = envs.step(actions)
        # next_states = next_states.reshape(num_envs, board.size_square)

        positions = np.random.randint(0, self.replay_buffer.capacity, num_envs)
        for position, last_obs, action, reward, obs, done in zip(
            positions, last_obss, actions, rewards, obss, dones
        ):
            if gamma == gamma_high and reward < -1.0:
                # likely the action was chosen at random, do not add to replay buffer
                continue
            self.replay_buffer.push(position, (last_obs, action, reward, obs, done))

        return obss

    async def train_on_buffer(self):
        loss = None
        gamma = gamma_low
        size = self.replay_buffer.size()
        if size > start_training:
            for i in range(epochs):
                batch = self.replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                gamma = calculate_gamma(np.array(rewards), np.array(dones))

                states = th.tensor(
                    np.array(states), dtype=th.float32, device=Device.XPU
                )
                actions = th.tensor(actions, dtype=th.float32, device=Device.XPU)
                if is_logit_env:
                    actions = actions.to(th.int64)
                rewards = th.tensor(rewards, dtype=th.float32, device=Device.XPU)
                next_states = th.tensor(
                    np.array(next_states), dtype=th.float32, device=Device.XPU
                )
                dones = th.tensor(dones, dtype=th.float32, device=Device.XPU)

                if is_logit_env:
                    loss = self.get_logit_loss(
                        states, next_states, rewards, dones, actions, gamma
                    )
                else:
                    loss = self.get_seq_loss(states, next_states, rewards, dones, gamma)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            del states, next_states, actions, rewards, dones
        else:
            print(f"{np.round(size / start_training * 100, 2)} %")
        return loss, gamma, self.optimizer_scheduler.get_last_lr()[0]

    @staticmethod
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
            dtype=FLOAT_TYPE,
        )

    @staticmethod
    def get_seq_actions(eps, q_values):
        actions = to_probs(q_values.to(Device.XPU).detach())
        for _ in range(int(num_envs * eps)):
            actions[np.random.randint(0, num_envs - 1)] = np.random.rand()
        return actions

    @staticmethod
    def get_logit_entropy(q_values):
        probs = th.nn.functional.softmax(q_values.clone().detach(), dim=-1)
        return -th.sum(probs * (probs + 1e-8).log(), dim=-1).mean()

    @staticmethod
    def get_seq_entropy(q_values):
        return th.std(q_values)

    def get_logit_loss(self, states, next_states, rewards, dones, actions, gamma):
        with th.no_grad():
            target_q_values = self.target_model(next_states)
            target = rewards + gamma * target_q_values.max(dim=1)[0] * (1 - dones)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(q_value, target)

        if th.any(th.isnan(loss)):
            print(q_value, target, actions, q_values, states)
            raise ValueError("NAN")
        return loss

    def get_seq_loss(self, states, next_states, rewards, dones, gamma):
        actions = self.model(states).squeeze(1)
        target_actions = self.target_model(next_states).squeeze(1).detach()
        target = rewards + gamma * target_actions * (1 - dones)

        return nn.functional.mse_loss(actions, target)

    def get_seq_advantage_loss(self, states, next_states, rewards, dones, gamma):
        actions = self.model(states).squeeze(1)
        target_actions = self.target_model(next_states).squeeze(1).detach()
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
    params = {
        "num_envs": num_envs,
        "num_workers": num_workers,
        "num_episodes": num_episodes,
        "learning_rate": learning_rate,
        "lr_shape": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "target_update": target_update_frequency,
        "buffer_size_low": buffer_size_low,
        "buffer_size_high": buffer_size_high,
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
    )

    is_logit_env = True
    envs = MultiprocessAsyncEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            SyncVectorEnv(
                [
                    lambda: Logit11GraphEnv(color=False, models=[None])
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
    dqn = DQN(envs)
    try:
        # with th.xpu.amp.autocast(enabled=True, dtype=TH_FLOAT_TYPE):
        # with th.amp.autocast(Device.XPU.value, enabled=True, dtype=th.bfloat16):
        dqn.run()
    finally:
        writer.close()

        if not os.path.exists("simple_dqn"):
            os.mkdir("simple_dqn")
        color = "white" if args.color else "black"
        th.save(dqn.model.state_dict(), f"simple_dqn/dqn-model-{color}.v{args.version}")
        th.save(
            dqn.target_model.state_dict(),
            f"simple_dqn/dqn-target-model-{color}.v{args.version}",
        )
        th.save(
            dqn.optimizer.state_dict(),
            f"simple_dqn/dqn-optimizer-{color}.v{args.version}",
        )
