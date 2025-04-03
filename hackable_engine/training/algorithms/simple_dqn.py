# -*- coding: utf-8 -*-
import asyncio
import os
from argparse import ArgumentParser
from collections import deque
from itertools import chain

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import FLOAT_TYPE, TH_FLOAT_TYPE
from hackable_engine.common.custom_threads import ReturningTargetThread
from hackable_engine.training.algorithms.util.replay_buffer import (
    ReplayBuffer,
    Experience,
)
from hackable_engine.training.constants import (
    LRShape,
    get_learning_rate_decay,
    TargetUpdateMode,
    GammaMode,
    BufferMode,
)
from hackable_engine.training.device import Device
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
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

# fmt: off
cnn_shape = (256, 512, 1024)
cnn_kernels = (5, 3, 3)
cnn_strides = (1, 1, 1, 1)
cnn_paddings = (0, 0, 0, 0)
GNN_SHAPES = {
    GraphSG: (54, 108, 216, 432, 324),  # 54 for 9 features, 36 for 3
    GraphRGCN: (54, 108, 216, 216, 144),
    GraphGAT: (54, 72, 90, 108, 144),
    GraphGIN: (54, 108, 216, 216, 144),  # maybe lower dim + faster run with 16 epochs & 2048 envs
    GraphGINE: (216, 216, 216, 216, 216),
}
# fmt: on

board_size = 13
env_class = Logit13GraphEnv
model_class = GraphSG
gnn_shape = GNN_SHAPES[model_class]
mlp_shape = (256,)  # board_size**2)
num_episodes = 1024
episode_env_steps = board_size**2
base_num_envs = 1024
"""relative for tensorboard graphs, such that training sessions are comparable"""
num_envs = 1026
num_workers = 6
lr_gnn = 2.1e-4
lr_mlp = 1e-4  # larger than the final lr_gnn
lr_gamma = 1e-4
lr_shape_gnn = LRShape.WARMUP_SIGMOID
lr_shape_mlp = LRShape.WARMUP_ONE
lr_warm_up_len = 0.18
batch_size = 64
gamma_mode = GammaMode.MANUAL
"""if set to TRAINED then below params are ignored"""
expected_episode_steps = 64
gamma_low = 0.1
gamma_high = (expected_episode_steps - 1) / expected_episode_steps
gamma_delay = 0
"""when mean rewards reach `gamma_delay` gamma becomes `gamma_high`, otherwise proportional to the rewards"""
assert (
    gamma_delay <= 0.66
)  # there is a custom formula for which larger delay will cause gamma to never reach maximum
epsilon_max = 0.5
epsilon_min = 0.005
target_update_frequency = 32
target_update_mode = TargetUpdateMode.SOFT
tau = 0.5
"""soft target update proportion"""
buffer_size_low = 1 * episode_env_steps * base_num_envs
buffer_size_high = 10 * episode_env_steps * base_num_envs
buffer_mode = BufferMode.RAM_AND_DISK
disk_buffer = 10 * buffer_size_high
start_training = buffer_size_low * 0.95
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
    hyperbolic = 1 / (1 + 99 * x**0.5)
    # exponential = 0.995 ** episode
    # return max((hyperbolic, exponential))
    return min(1, hyperbolic)


class DQN:
    def __init__(self, envs, color, load_version: int | None = None):
        self.envs = envs
        self.model = (
            model_class(  # does best with high learning rates like 6e-4
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
                # use_res=False,
            )
            .to(Device.XPU)
            .to(th.float32)
        )
        self.target_model = (
            model_class(
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
                # use_res=False,
            )
            .to(Device.XPU)
            .to(th.float32)
        )
        self.tmp_loss = th.tensor(0, dtype=TH_FLOAT_TYPE, device=Device.XPU)

        weight_decay = 2e-5 if model_class is GraphSG else 0
        """use weight_decay for models which tend to have growing gradient towards the end"""
        optimizer_params = [
            {
                "params": [
                    param for layer in self.model.gnn for param in layer.parameters()
                ],
                "lr": lr_gnn,
            },
            {
                "params": [
                    param for layer in self.model.mlp for param in layer.parameters()
                ],
                "lr": lr_mlp,
            },
        ]
        scheduler_params = [
            get_learning_rate_decay(lr_shape_gnn, num_episodes, lr_warm_up_len),
            get_learning_rate_decay(lr_shape_mlp, num_episodes, lr_warm_up_len),
        ]
        if gamma_mode is GammaMode.TRAINED:
            self.model.gamma = nn.Parameter(
                th.tensor(-2, dtype=TH_FLOAT_TYPE, device=Device.XPU)
            )
            self.target_model.gamma = nn.Parameter(
                th.tensor(-2, dtype=TH_FLOAT_TYPE, device=Device.XPU)
            )
            optimizer_params.append({"params": self.model.gamma, "lr": lr_gamma})
            scheduler_params.append(
                get_learning_rate_decay(LRShape.ONE, num_episodes, lr_warm_up_len)
            )

        # optimizer = optim.SGD(
        #     model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
        # )
        self.optimizer = optim.Adam(
            optimizer_params,
            weight_decay=weight_decay,
        )
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size_low, storage_type=ReplayBuffer.StorageType.LIST
        )
        self.optimizer_scheduler = LambdaLR(
            self.optimizer,
            scheduler_params,
        )

        if load_version:
            self.model.load_state_dict(
                th.load(
                    f"simple_dqn/dqn-model-{color}.v{load_version}", weights_only=True
                )
            )
            self.target_model.load_state_dict(
                th.load(
                    f"simple_dqn/dqn-target-model-{color}.v{load_version}",
                    weights_only=True,
                )
            )
            self.optimizer.load_state_dict(
                th.load(
                    f"simple_dqn/dqn-optimizer-{color}.v{load_version}",
                    weights_only=True,
                )
            )

        th._dynamo.reset()
        self.model = th.compile(self.model)
        self.target_model = th.compile(self.target_model)

    def run(self, version: int):
        self.replay_buffer.setup_disk_backup(
            disk_buffer, batch_size, self.envs.single_observation_space.shape, version
        )
        asyncio.run(self.train())

    async def train(self):
        eps = epsilon_max
        gamma = gamma_low
        target_update_f = target_update_frequency / 2
        target_update_m = TargetUpdateMode.HARD

        obs: list[np.ndarray] = self.envs.reset()
        old_experiences = ()
        new_experiences = ()

        for episode in range(0, num_episodes, episode_step):
            print("episode", episode, "buffer", self.replay_buffer.size())
            steps = 0
            while True:
                steps += 1
                q_values = self.model(th.from_numpy(np.stack(obs, 0)).to(Device.XPU))

                if is_logit_env:
                    actions = self.get_logit_actions(eps, q_values)
                else:
                    actions = self.get_seq_actions(eps, q_values)
                # action_queue.extend(actions.flatten().tolist())

                # t0 = perf_counter()
                # obs = await self.run_env(obs, actions)
                # print("env takes", perf_counter() - t0)
                #
                # t0 = perf_counter()
                # loss, gamma = await self.train_on_buffer()
                # print("training takes", perf_counter() - t0)
                (obs, new_experiences), (loss, gamma, old_experiences) = await asyncio.gather(
                    self.run_env(obs, actions),
                    self.train_on_buffer(gamma, new_experiences, old_experiences),
                )

                if loss and steps > episode_env_steps:
                    value_loss_queue.append(loss)

                    if is_logit_env:
                        entropy = self.get_logit_entropy(q_values)
                    else:
                        entropy = self.get_seq_entropy(q_values)
                    entropy_queue.append(entropy.item())

                    break

            eps = epsilon_max * epsilon_decay(episode) + epsilon_min
            for i in range(episode_step):
                self.optimizer_scheduler.step()

            if episode % target_update_f == 2:
                print("updating target")
                self.update_target(target_update_m)
                if episode > target_update_f:
                    target_update_f = target_update_frequency
                    target_update_m = target_update_mode
                    if gamma > 2 * gamma_low:
                        self.replay_buffer.capacity = buffer_size_high
                        if buffer_mode is BufferMode.RAM_AND_DISK:
                            self.replay_buffer.backup.active = True

            if episode % 1 == 0:
                gnn_gradients = []
                for param in [
                    param for layer in self.model.gnn for param in layer.parameters()
                ]:
                    if param.grad is not None:
                        gnn_gradients.append(param.grad.view(-1))
                gnn_gradients = th.cat(gnn_gradients)
                gnn_gradient = gnn_gradients.norm()
                del gnn_gradients

                mlp_gradients = []
                for param in [
                    param for layer in self.model.mlp for param in layer.parameters()
                ]:
                    if param.grad is not None:
                        mlp_gradients.append(param.grad.view(-1))
                mlp_gradients = th.cat(mlp_gradients)
                mlp_gradient = mlp_gradients.norm()
                del mlp_gradients

                lr_gnn_, lr_mlp_ = self.optimizer_scheduler.get_last_lr()[:2]
                log_progress(
                    episode, gnn_gradient, mlp_gradient, eps, gamma, lr_gnn_, lr_mlp_
                )
                # log_progress(episode, gradient, eps, gamma.item(), lr_gnn_, lr_mlp_)

    async def run_env(
        self, last_obss, actions
    ) -> tuple[np.array, tuple[Experience, ...]]:
        obss, rewards, dones, _, _ = await envs.step(actions)

        experiences = []
        experience_count = 0
        positions = np.random.randint(0, self.replay_buffer.capacity, num_envs)
        for position, last_obs, action, reward, obs, done in zip(
            positions, last_obss, actions, rewards, obss, dones
        ):
            # if gamma == gamma_high and reward < -1.0:
            #     # likely the action was chosen at random, do not add to replay buffer
            #     continue
            experience = (last_obs, action, reward, obs, done)
            if experience_count < batch_size:
                experiences.append(experience)
                experience_count += 1
            self.replay_buffer.push(position, experience)

        return obss, tuple(experiences)

    async def train_on_buffer(
        self,
        last_gamma,
        new_experiences: tuple[Experience, ...],
        old_experiences: tuple[Experience, ...],
    ) -> tuple[float, float, tuple[Experience, ...]]:
        prev_loss: float = 0
        gamma = last_gamma
        size = self.replay_buffer.size()
        if size > start_training:
            # old_experiences = self.replay_buffer.sample(batch_size * (epochs - 1))
            states, actions, rewards, next_states, dones = zip(
                *chain(new_experiences, old_experiences)
            )

            next_states = th.from_numpy(np.array(next_states, dtype=FLOAT_TYPE))
            next_states = next_states.pin_memory().to(
                device=Device.XPU, non_blocking=True
            )
            thread = ReturningTargetThread(
                target=self.target_model, args=(next_states,)
            )
            thread.start()

            next_experiences = tuple(self.replay_buffer.sample(batch_size * (epochs - 1)))

            states = th.from_numpy(np.array(states, dtype=FLOAT_TYPE))
            # if not is_logit_env: then float32 not int64
            actions = th.from_numpy(np.array(actions, dtype=np.int64))
            rewards = th.from_numpy(np.array(rewards, dtype=FLOAT_TYPE))
            dones = th.from_numpy(np.array(dones, dtype=FLOAT_TYPE))

            gamma = self.calculate_gamma(rewards, dones, last_gamma)

            states = states.pin_memory().to(device=Device.XPU, non_blocking=True)
            actions = actions.pin_memory().to(device=Device.XPU, non_blocking=True)
            rewards = rewards.pin_memory().to(device=Device.XPU, non_blocking=True)
            dones = dones.pin_memory().to(device=Device.XPU, non_blocking=True)

            states_chunks = states.chunk(epochs, dim=0)
            actions_chunks = actions.chunk(epochs, dim=0)
            rewards_chunks = rewards.chunk(epochs, dim=0)
            dones_chunks = dones.chunk(epochs, dim=0)

            # copying loss value from gpu takes significant time so let's do it while target thread is running
            prev_loss = self.tmp_loss.item() / epochs
            self.tmp_loss.zero_()

            target_q_values = thread.join().detach()
            # target_q_values = self.target_model(next_states).detach()
            target_q_values_chunks = target_q_values.chunk(epochs, dim=0)

            for i in range(epochs):
                states_ = states_chunks[i]
                actions_ = actions_chunks[i]
                rewards_ = rewards_chunks[i]
                target_q_values_ = target_q_values_chunks[i]
                dones_ = dones_chunks[i]

                if is_logit_env:
                    loss = self.get_logit_loss(
                        states_, target_q_values_, rewards_, dones_, actions_, gamma
                    )
                else:
                    loss = self.get_seq_loss(
                        states_, target_q_values_, rewards_, dones_, gamma
                    )
                del states_, actions_, rewards_, target_q_values_, dones_

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.tmp_loss += loss.detach()
            del (
                old_experiences,
                new_experiences,
                states,
                next_states,
                actions,
                rewards,
                dones,
                target_q_values,
            )
        else:
            if size >= batch_size * epochs:
                next_experiences = tuple(self.replay_buffer.sample(batch_size * (epochs - 1)))
            else:
                next_experiences = ()
            print(f"{np.round(size / start_training * 100, 2)} %")
        return prev_loss, gamma, next_experiences

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

    def get_logit_loss(
        self, states, target_q_values, rewards, dones, actions, gamma: th.Tensor | float
    ):
        target = rewards + gamma * target_q_values.max(dim=1)[0] * (1 - dones)
        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        return nn.functional.mse_loss(q_value, target)

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

    def update_target(self, target_update_m):
        if target_update_m is TargetUpdateMode.HARD:
            self.target_model.load_state_dict(self.model.state_dict())
        elif target_update_m is TargetUpdateMode.SOFT:
            for target_param, model_param in zip(
                self.target_model.parameters(), self.model.parameters()
            ):  # TODO: is this copy a source of memory leak?
                target_param.data.copy_(
                    (1 - tau) * target_param.data + tau * model_param.data
                )
        else:
            raise ValueError("unknown target update mode")

    def calculate_gamma(self, rewards, dones, last_gamma) -> th.Tensor | float:
        if gamma_mode is GammaMode.MANUAL:
            sum_reward = (rewards * dones).sum().item()
            if sum_reward == 0:
                return last_gamma

            mean_reward = sum_reward / dones.sum().item()
            reward_curve_point = (
                -(mean_reward**2) / 3 + mean_reward / 2 + 1 / 3
                if mean_reward < 0.75
                else 0.66
            )
            a = min(max(gamma_delay - reward_curve_point, 0), 1)
            """starts from 1, gets closer to 0 as rewards grow, ==0 from reward>0.5, >0.5 from reward>-0.5"""
            gamma = gamma_low * a + gamma_high * (1 - a)
            return (gamma + last_gamma) / 2
        elif gamma_mode is GammaMode.TRAINED:
            return nn.functional.sigmoid(self.model.gamma)

        raise ValueError("unknown gamma mode")


def to_probs(actions):
    return nn.functional.sigmoid(actions)
    # mean = actions.mean()
    # std = actions.std()
    # return (nn.functional.tanh((actions - mean)/(std + 1e-8)) + 1) / 2


def log_progress(episode, gnn_gradient, mlp_gradient, eps, gamma, lr_gnn_, lr_mlp_):
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
    writer.add_scalar("misc/gradient", gnn_gradient, episode)
    writer.add_scalar("misc/mlp_gradient", mlp_gradient, episode)
    writer.add_scalar("misc/lr_gnn", lr_gnn_, episode)
    writer.add_scalar("misc/lr_mlp", lr_mlp_, episode)


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
        "-l",
        "--load-version",
        type=int,
        help="version of the model to load",
        required=False,
    )
    parser.add_argument(
        "-v", "--version", type=int, help="version of the model to save", required=True
    )
    parser.add_argument(
        "-c",
        "--color",
        type=int,
        required=True,
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
        "lr_gnn": lr_gnn,
        "lr_mlp": lr_mlp,
        "lr_shape_gnn": lr_shape_gnn,
        "lr_shape_mlp": lr_shape_mlp,
        "lr_warm_up_len": lr_warm_up_len,
        "batch_size": batch_size,
        "epochs": epochs,
        "target_update_freq": target_update_frequency,
        "target_update_mode": target_update_mode,
        "tau": tau,
        "gamma_mode": gamma_mode,
        "gamma_low": gamma_low,
        "gamma_high": gamma_high,
        "gamma_delay": gamma_delay,
        "buffer_size_low": buffer_size_low,
        "buffer_size_high": buffer_size_high,
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
    )

    is_logit_env = True
    envs = MultiprocessAsyncEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            SyncVectorEnv(
                [
                    lambda: env_class(color=False, models=[None])
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
    color = "white" if args.color else "black"
    dqn = DQN(envs, color, args.load_version)
    try:
        # with th.xpu.amp.autocast(enabled=True, dtype=TH_FLOAT_TYPE):
        dqn.run(args.version)
    finally:
        writer.close()

        if not os.path.exists("simple_dqn"):
            os.mkdir("simple_dqn")
        th.save(dqn.model.state_dict(), f"simple_dqn/dqn-model-{color}.v{args.version}")
        th.save(
            dqn.target_model.state_dict(),
            f"simple_dqn/dqn-target-model-{color}.v{args.version}",
        )
        th.save(
            dqn.optimizer.state_dict(),
            f"simple_dqn/dqn-optimizer-{color}.v{args.version}",
        )
