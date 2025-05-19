# -*- coding: utf-8 -*-
import asyncio
import os
from argparse import ArgumentParser
from collections import deque
from itertools import chain
from random import choice

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import intel_extension_for_pytorch

from hackable_engine.board.hex.bitboard_utils import (
    int_to_binary_float_array,
    generate_cells,
)
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
    GraphSG: (54, 108, 216, 324, 432, 486),
    # GraphSG: (54, 162, 486, 486, 486, 486),
    # GraphSG: (54, 162, 486, 648, 648, 648),
    # GraphGIN: (54, 108, 216, 432, 432, 432),
    GraphGIN: (54, 162, 486, 486, 486, 486),
    GraphGINE: (54, 108, 216, 432, 432, 432),
    GraphRGCN: (54, 108, 216, 324, 324, 324),
    GraphGAT: (54, 72, 90, 108, 216),
}
# fmt: on

board_size = 13
board_size_squared = board_size**2
binary_one = 2**board_size_squared - 1
env_class = Logit13GraphEnv
model_class = GraphSG
allow_illegal = True
gnn_shape = GNN_SHAPES[model_class]
mlp_shape = (256,)  # board_size**2)
num_episodes = 1024
episode_env_steps = board_size  # **2
base_num_envs = 128
"""relative for tensorboard graphs, such that training sessions are comparable"""
num_envs = 128
num_workers = 8
lr_gnn = 1.4e-4
lr_mlp = 2.1e-4  # if shape ONE, larger than the final lr_gnn
lr_gamma = 1e-4
lr_shape_gnn = LRShape.ONE
lr_shape_mlp = LRShape.ONE
lr_warm_up_len = 0.12
target_update_frequency = 1
target_update_mode = TargetUpdateMode.SOFT
tau = 0.005  # recommended 0.005 – 0.01 for a learning rate of 1e-3, lower for smaller learning rates
"""soft target update proportion"""
batch_size = 64
gamma_mode = GammaMode.RETRAINED
"""if set to TRAINED then below params are ignored"""
expected_episode_steps = 64
gamma_low = 0.1
gamma_high = (expected_episode_steps - 1) / expected_episode_steps
gamma_delay = 0
"""when mean rewards reach `gamma_delay` gamma becomes `gamma_high`, otherwise proportional to the rewards"""
assert (
    gamma_delay <= 0.66
)  # there is a custom formula for which larger delay will cause gamma to never reach maximum
epsilon_max = 0.6
epsilon_min = 0.02
control_weight_param = 0.5
legality_weight_param = 0.25 if allow_illegal else 0.05
entropy_weight_param = 0.01
entropy_temperature = 1.0
"""
If Q-values are sharply peaked (e.g., one action always dominates), increase it (e.g., 2.0 or 5.0) to "soften" the distribution and get a smoother entropy measure.
If Q-values are very flat, you can reduce it to make entropy more sensitive to small preference differences.
"""
should_load_init_buffer = True
should_dump_init_buffer = False
buffer_priority_rate_max = 1.0
buffer_priority_rate_min = 1.0
buffer_size_low = 100_000
buffer_size_high = 2_000_000
buffer_mode = BufferMode.RAM
disk_buffer = 4 * buffer_size_high
start_training = buffer_size_low * 0.95
buffer_ratio = 16
"""Defines the proportion between fresh experience training and buffer training"""
epochs = num_envs * buffer_ratio // batch_size
episode_step = num_envs // base_num_envs
assert episode_step >= 1

# action_queue = deque(maxlen=board_size**2 * num_envs)
# max_action_queue = deque(maxlen=num_envs)
# policy_loss_queue = deque(maxlen=2 * num_envs)
value_loss_queue = deque(maxlen=2 * num_envs)
control_loss_queue = deque(maxlen=2 * num_envs)
legality_loss_queue = deque(maxlen=2 * num_envs)
entropy_queue = deque(maxlen=2 * num_envs)


def epsilon_decay(episode):
    x = episode / num_episodes
    hyperbolic = 1 / (1 + 99 * x**0.5)
    # exponential = 0.995 ** episode
    # return max((hyperbolic, exponential))
    return min(1, hyperbolic)


def buffer_priority_rate_decay(episode):
    x = episode / num_episodes
    hyperbolic = 1 / (1 + 15 * x**2)
    return min(1, hyperbolic)


class DQN:
    def __init__(
        self, envs, color: bool, version: int, load_version: int | None = None
    ):
        self.envs = envs
        self.color = color
        self.version = version
        self.model = (
            model_class(  # does best with high learning rates like 6e-4
                node_count=board.size_square,
                node_features=9,
                output_size=1,
                batch_size=batch_size,
                num_envs=num_envs,
                # num_epochs=epochs,
                gnn_shape=gnn_shape,
                # gnn_heads=6,
                mlp_shape=mlp_shape,
                edge_index=board.edge_index,
                # edge_types=board.edge_types,
                # use_res=True,
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
                # num_epochs=epochs,
                gnn_shape=gnn_shape,
                # gnn_heads=6,
                mlp_shape=mlp_shape,
                edge_index=board.edge_index,
                # edge_types=board.edge_types,
                # use_res=True,
            )
            .to(Device.XPU)
            .to(th.float32)
        )
        self.target_model.train(False)

        self.tmp_value_loss = th.tensor(0, dtype=TH_FLOAT_TYPE, device=Device.XPU)
        self.tmp_control_loss = th.tensor(0, dtype=TH_FLOAT_TYPE, device=Device.XPU)
        self.tmp_legality_loss = th.tensor(0, dtype=TH_FLOAT_TYPE, device=Device.XPU)
        self.tmp_entropy = th.tensor(0, dtype=TH_FLOAT_TYPE, device=Device.XPU)

        weight_decay = 0  # 2e-5 if model_class is GraphSG else 0
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
            board_size=board_size,
            capacity=buffer_size_low,
            storage_type=ReplayBuffer.StorageType.LIST,
            priority_rate=buffer_priority_rate_max,
        )
        self.optimizer_scheduler = LambdaLR(
            self.optimizer,
            scheduler_params,
        )

        if load_version is not None:
            color_name = "white" if self.color else "black"
            model_weights = th.load(
                f"dqn/dqn-model-{color_name}.v{load_version}", weights_only=True
            )
            self.model.load_state_dict(
                {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}
            )
            target_weights = th.load(
                f"dqn/dqn-target-model-{color_name}.v{load_version}",
                weights_only=True,
            )
            self.target_model.load_state_dict(
                {k.replace("_orig_mod.", ""): v for k, v in target_weights.items()}
            )
            optimizer_weights = th.load(
                f"dqn/dqn-optimizer-{color_name}.v{load_version}",
                weights_only=True,
            )
            self.optimizer.load_state_dict(
                {k.replace("_orig_mod.", ""): v for k, v in optimizer_weights.items()}
            )
            print(f"weights from color {color_name} version {load_version} loaded")
        else:
            self.target_model.load_state_dict(self.model.state_dict())

        # th._dynamo.reset()
        # self.model = th.compile(self.model)
        # self.target_model = th.compile(self.target_model)

        self.obs: list[np.ndarray] = []
        self.black_masks: list[int] = []
        self.white_masks: list[int] = []
        self.training_started: bool = False
        self.buffer_loaded: bool = False

    def run(self, version: int):
        if buffer_mode in [BufferMode.DISK, BufferMode.RAM_AND_DISK]:
            self.replay_buffer.setup_disk_backup(
                disk_buffer,
                batch_size,
                self.envs.single_observation_space.shape,
                version,
            )
        asyncio.run(self.train())

    async def train(self):
        eps = epsilon_max
        gamma = gamma_low
        target_update_f = (
            target_update_frequency // 2
            if target_update_frequency >= 8
            else target_update_frequency
        )
        target_update_m = TargetUpdateMode.HARD

        if not allow_illegal:
            self.replay_buffer.capacity = buffer_size_high

        self.obs: list[np.ndarray]
        self.black_masks: tuple[int, ...]
        self.white_masks: tuple[int, ...]
        self.obs, self.black_masks, self.white_masks = self.envs.reset()
        old_experiences = ()
        new_experiences = ()

        ep_start = 0
        for i in range(ep_start):
            self.optimizer_scheduler.step()

        for episode in range(
            ep_start, num_episodes, episode_step
        ):  # TODO: start from 0 again
            print("episode", episode, "buffer", self.replay_buffer.size())
            steps = 0
            while True:
                steps += 1
                q_values, _, legality_prediction = self.model(
                    th.from_numpy(np.stack(self.obs, 0)).to(Device.XPU)
                )

                if is_logit_env:
                    actions = self.get_logit_actions(eps, q_values, legality_prediction)
                else:
                    actions = self.get_seq_actions(eps, q_values)
                # action_queue.extend(actions.flatten().tolist())

                # t0 = perf_counter()
                # self.obs, self.illegal_masks, new_experiences = await self.run_env(self.obs, actions)
                # print("env takes", perf_counter() - t0)
                #
                # t0 = perf_counter()
                # value_loss, legality_loss, entropy, gamma, old_experiences = await self.train_on_buffer(gamma, new_experiences, old_experiences)
                # print("training takes", perf_counter() - t0)
                (self.obs, self.black_masks, self.white_masks, new_experiences), (
                    value_loss,
                    control_loss,
                    legality_loss,
                    entropy,
                    gamma,
                    old_experiences,
                ) = await asyncio.gather(
                    self.run_env(self.obs, actions),
                    self.train_on_buffer(gamma, new_experiences, old_experiences),
                )

                if value_loss and steps > episode_env_steps:
                    value_loss_queue.append(value_loss)
                    control_loss_queue.append(control_loss)
                    legality_loss_queue.append(legality_loss)
                    entropy_queue.append(entropy)

                    # if is_logit_env:
                    #     entropy = self.get_logit_entropy(q_values)
                    # else:
                    #     entropy = self.get_seq_entropy(q_values)
                    # entropy_queue.append(entropy.item())

                    break

            if self.replay_buffer.size() < start_training:
                continue

            eps = (epsilon_max - epsilon_min) * epsilon_decay(episode) + epsilon_min
            decay = (buffer_priority_rate_max - buffer_priority_rate_min) * buffer_priority_rate_decay(episode)
            self.replay_buffer.priority_rate = decay + buffer_priority_rate_min
            for i in range(episode_step):
                self.optimizer_scheduler.step()

            remainder = 1 if target_update_frequency == 2 else 2
            if episode % target_update_f == remainder or target_update_frequency == 1:
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
                q_mean = q_values.mean().item()
                q_std = q_values.std().item()

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
                    episode,
                    q_mean,
                    q_std,
                    gnn_gradient,
                    mlp_gradient,
                    eps,
                    gamma,
                    lr_gnn_,
                    lr_mlp_,
                    self.replay_buffer.priority_rate,
                )
                # log_progress(episode, gradient, eps, gamma.item(), lr_gnn_, lr_mlp_)
            if episode % 64 == 0 and episode > 0:
                self.save_checkpoint(episode)

    async def run_env(
        self, last_obss, actions
    ) -> tuple[np.array, tuple[int, ...], tuple[int, ...], tuple[Experience, ...]]:
        obss, rewards, dones, _, black_masks, white_masks = await envs.step(actions)

        experiences: list[Experience] = []
        experience_count = 0
        positions = np.random.randint(0, self.replay_buffer.capacity, num_envs)
        for (
            position,
            last_obs,
            action,
            reward,
            obs,
            done,
            black_mask,
            white_mask,
        ) in zip(
            positions,
            last_obss,
            actions,
            rewards,
            obss,
            dones,
            black_masks,
            white_masks,
        ):
            # if gamma == gamma_high and reward < -1.0:
            #     # likely the action was chosen at random, do not add to replay buffer
            #     continue
            experience = (last_obs, action, reward, obs, done, black_mask, white_mask)
            experiences.append(experience)
            # if experience_count < batch_size:
            #     experiences.append(experience)
            #     experience_count += 1
            self.replay_buffer.push(position, experience)

        return obss, black_masks, white_masks, tuple(experiences)

    async def train_on_buffer(
        self,
        last_gamma,
        new_experiences: tuple[Experience, ...],
        old_experiences: tuple[Experience, ...],
    ) -> tuple[float, float, float, float, float, tuple[Experience, ...]]:
        prev_value_loss: float = 0
        prev_control_loss: float = 0
        prev_legality_loss: float = 0
        prev_entropy: float = 0
        gamma = last_gamma
        size = self.replay_buffer.size()
        if size > start_training:
            if not self.training_started and should_dump_init_buffer:
                self.training_started = True
                await self.replay_buffer.dump_to_disk(
                    self.envs.single_observation_space.shape
                )
                print("dumped initial buffer to disk")

            if not old_experiences:
                old_experiences = tuple(
                    self.replay_buffer.sample(num_envs * (buffer_ratio - 1))
                )

            states, actions, rewards, next_states, dones, black_masks, white_masks = (
                zip(*chain(new_experiences, old_experiences))
            )

            next_states = th.from_numpy(np.array(next_states, dtype=FLOAT_TYPE))
            next_states = next_states.pin_memory().to(
                device=Device.XPU, non_blocking=True
            )

            thread = ReturningTargetThread(
                target=self.run_target_model,
                args=(
                    self.target_model,
                    next_states,
                ),
            )
            thread.start()

            illegality_masks = []
            control_stats_list = []
            for black_mask, white_mask in zip(black_masks, white_masks):
                illegality_masks.append(
                    int_to_binary_float_array(
                        black_mask | white_mask, board_size_squared
                    )
                )
                control_stats_list.append(
                    self.replay_buffer.find_closest_prob_oc_stats(
                        black_mask, white_mask
                    )
                )

            # illegality_masks = [
            #     int_to_binary_float_array(black_mask | white_mask, board_size_squared)
            #     for black_mask, white_mask in zip(black_masks, white_masks)
            # ]
            illegality_masks_stacked = th.from_numpy(np.stack(illegality_masks))
            control_stats_stacked = th.from_numpy(np.stack(control_stats_list))

            next_experiences = tuple(
                self.replay_buffer.sample(num_envs * (buffer_ratio - 1))
            )

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
            illegality_masks_stacked = illegality_masks_stacked.pin_memory().to(
                device=Device.XPU, non_blocking=True
            )
            control_stats_stacked = control_stats_stacked.pin_memory().to(
                device=Device.XPU, non_blocking=True
            )

            states_chunks = states.chunk(epochs, dim=0)
            actions_chunks = actions.chunk(epochs, dim=0)
            rewards_chunks = rewards.chunk(epochs, dim=0)
            dones_chunks = dones.chunk(epochs, dim=0)
            illegality_masks_chunks = illegality_masks_stacked.chunk(epochs, dim=0)
            control_stats_chunks = control_stats_stacked.chunk(epochs, dim=0)

            # copying loss value from gpu takes significant time so let's do it while target thread is running
            prev_value_loss = self.tmp_value_loss.item() / epochs
            prev_control_loss = self.tmp_control_loss.item() / epochs
            prev_legality_loss = self.tmp_legality_loss.item() / epochs
            prev_entropy = self.tmp_entropy.item() / epochs
            control_loss_weight = (
                abs(prev_value_loss / prev_control_loss) if prev_control_loss else 1.0
            )
            legality_loss_weight = (
                abs(prev_value_loss / prev_legality_loss) if prev_legality_loss else 1.0
            )
            entropy_weight = (
                -abs(prev_value_loss / prev_entropy) if prev_entropy else -1.0
            )
            self.tmp_value_loss.zero_()
            self.tmp_control_loss.zero_()
            self.tmp_legality_loss.zero_()
            self.tmp_entropy.zero_()

            target_q_values = thread.join()
            # target_q_values = target_q_values.detach()

            # target_q_values = self.target_model(next_states).detach()
            target_q_values_chunks = target_q_values.chunk(epochs, dim=0)

            for i in range(epochs):
                states_ = states_chunks[i]
                actions_ = actions_chunks[i]
                rewards_ = rewards_chunks[i]
                target_q_values_ = target_q_values_chunks[i]
                dones_ = dones_chunks[i]
                illegality_masks = illegality_masks_chunks[i]
                control_stats = control_stats_chunks[i]

                if is_logit_env:
                    value_loss, legality_loss, control_loss, entropy = (
                        self.get_logit_loss(
                            states_,
                            target_q_values_,
                            rewards_,
                            dones_,
                            actions_,
                            gamma,
                            illegality_masks,
                            control_stats,
                        )
                    )
                else:
                    value_loss, legality_loss, control_loss, entropy = (
                        self.get_seq_loss(
                            states_, target_q_values_, rewards_, dones_, gamma
                        )
                    )
                del states_, actions_, rewards_, target_q_values_, dones_

                self.optimizer.zero_grad()
                (
                    value_loss
                    + control_loss * control_loss_weight * control_weight_param
                    + legality_loss * legality_loss_weight * legality_weight_param
                    + entropy * entropy_weight * entropy_weight_param
                ).backward()
                # clip_grad_norm_(self.model.parameters(), 1000)
                self.optimizer.step()
                self.tmp_value_loss += value_loss.detach()
                self.tmp_control_loss += control_loss.detach()
                self.tmp_legality_loss += legality_loss.detach()
                self.tmp_entropy += entropy.detach()
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
            if not self.buffer_loaded and should_load_init_buffer:
                self.buffer_loaded = True
                await self.replay_buffer.load_init_buffer(
                    self.envs.single_observation_space.shape, buffer_size_low
                )
                size = self.replay_buffer.size()
                print("initial buffer loaded")
            if size >= num_envs * (buffer_ratio - 1):
                next_experiences = tuple(
                    self.replay_buffer.sample(num_envs * (buffer_ratio - 1))
                )
            else:
                next_experiences = ()
            print(f"{np.round(size / start_training * 100, 2)} %")

        return (
            prev_value_loss,
            prev_control_loss,
            prev_legality_loss,
            prev_entropy,
            gamma,
            next_experiences,
        )

    @staticmethod
    def run_target_model(target_model, states):
        try:
            chunks = states.chunk(2)
            # print("calculating target model...")
            # t0 = perf_counter()
            with th.no_grad():
                q_values_chunks = [target_model(chunk).detach() for chunk in chunks]
            # print("target model calculated in", perf_counter() - t0)
            return th.cat(q_values_chunks)
        except Exception as e:
            print("*************\n", e)

    def get_logit_actions(self, eps, q_values, legality_prediction):
        if allow_illegal:
            predicted_legality_mask = legality_prediction > 0.1
            legal_squares = th.nonzero(predicted_legality_mask)
            # TODO: still mask logits, but using legality prediction from the model, hard or soft
            # actions = []
            # for i in range(num_envs):
            #     predicted_legality_mask = legality_prediction[i] > 0.1
            #     legal_squares = th.nonzero(predicted_legality_mask, as_tuple=True)
            #     if np.random.rand() > eps:
            #         actions.append(choice(legal_squares))
            #     else:
            #         q_value = q_values[i]
            #         min_val = q_value.argmin().item()
            #         q_value[legal_squares] = q_value[min_val].item() - 1e-8
            #         actions.append(q_value.argmax().item())
            #
            # return np.array(actions, dtype=FLOAT_TYPE)

            # soft masking by multiplication times probability
            # softmasked_q_values = th.softmax(q_values, dim=-1) * legality_prediction
            # return np.array(
            #     [
            #         (
            #             np.random.choice(th.nonzero(predicted_legality_mask[i]).flatten())
            #             if np.random.rand() < eps
            #             else softmasked_q_values[i].argmax().item()
            #         )
            #         for i in range(num_envs)
            #     ],
            #     dtype=FLOAT_TYPE,
            # )

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
        else:
            actions = []
            for i, (black_mask, white_mask) in enumerate(
                zip(self.black_masks, self.white_masks)
            ):
                illegal_mask = black_mask | white_mask
                if np.random.rand() > eps:
                    legal_squares = list(generate_cells(binary_one ^ illegal_mask))
                    actions.append(choice(legal_squares))
                else:
                    illegal_squares = list(generate_cells(illegal_mask))
                    q_value = q_values[i]
                    min_val = q_value.argmin().item()
                    q_value[illegal_squares] = q_value[min_val].item() - 1e-8
                    actions.append(q_value.argmax().item())

            return np.array(actions, dtype=FLOAT_TYPE)

    @staticmethod
    def get_seq_actions(eps, q_values):
        actions = to_probs(q_values.to(Device.XPU).detach())
        for _ in range(int(num_envs * eps)):
            actions[np.random.randint(0, num_envs - 1)] = np.random.rand()
        return actions

    @staticmethod
    def get_logit_entropy(q_values):
        probs = F.softmax(q_values.clone().detach(), dim=-1)
        return -th.sum(probs * (probs + 1e-8).log(), dim=-1).mean()

    @staticmethod
    def get_seq_entropy(q_values):
        return th.std(q_values)

    def get_logit_loss(
        self,
        states,
        target_q_values,
        rewards,
        dones,
        actions,
        gamma: th.Tensor | float,
        illegality_masks: th.Tensor,
        control_stats: th.Tensor,
    ):
        target = rewards + gamma * target_q_values.max(dim=1)[0] * (1 - dones)
        q_values, control, legality = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # TODO: monitor the following values, periodically if heavy
        # td_error = th.abs(q_value - target).mean().item()
        # q_drift = F.mse_loss(q_values, target_q_values, reduction='mean').item()

        # control_log_probs = (F.softmax(control, dim=-1) + 1e-8).log()
        control_log_probs = F.log_softmax(control, dim=-1)
        # control_loss = -(control_log_probs * control_stats).mean()
        control_loss = F.kl_div(control_log_probs, control_stats, reduction="batchmean")

        q_probs = F.softmax(q_values / entropy_temperature, dim=-1)
        entropy = -(q_probs * (q_probs + 1e-8).log()).sum(dim=-1).mean()

        return (
            F.mse_loss(q_value, target),
            F.binary_cross_entropy(F.sigmoid(legality), illegality_masks),
            control_loss,
            entropy,
        )

    def get_seq_loss(self, states, next_states, rewards, dones, gamma):
        actions = self.model(states).squeeze(1)
        target_actions = self.target_model(next_states).squeeze(1).detach()
        target = rewards + gamma * target_actions * (1 - dones)

        return F.mse_loss(actions, target)

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
                    (1 - tau) * target_param.data + tau * model_param.data,
                    non_blocking=True,
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
            return F.sigmoid(self.model.gamma)
        elif gamma_mode is GammaMode.RETRAINED:
            return gamma_high

        raise ValueError("unknown gamma mode")

    def save_checkpoint(self, episode: int | None = None):
        color_name = "white" if self.color else "black"
        version = self.version if episode is None else f"{self.version}.{episode}"

        if not os.path.exists("dqn"):
            os.mkdir("dqn")
        th.save(self.model.state_dict(), f"dqn/dqn-model-{color_name}.v{version}")
        th.save(
            self.target_model.state_dict(),
            f"dqn/dqn-target-model-{color_name}.v{version}",
        )
        th.save(
            self.optimizer.state_dict(),
            f"dqn/dqn-optimizer-{color_name}.v{version}",
        )


def to_probs(actions):
    return F.sigmoid(actions)
    # mean = actions.mean()
    # std = actions.std()
    # return (F.tanh((actions - mean)/(std + 1e-8)) + 1) / 2


def log_progress(
    episode, q_mean, q_std, gnn_gradient, mlp_gradient, eps, gamma, lr_gnn_, lr_mlp_, priority_rate
):
    # actions = np.array(action_queue)
    # writer.add_histogram(
    #     "actions/raw",
    #     actions,
    #     bins="auto",
    #     max_bins=100,
    # )
    try:
        env_progress_data: EnvProgressData = envs.get_progress_data()
    except ZeroDivisionError:
        print("Zero division error")  # FIXME
        return
    writer.add_scalar("episode/length", env_progress_data.length_mean, episode)
    writer.add_scalar("episode/win_length", env_progress_data.win_length_mean, episode)
    writer.add_scalar(
        "episode/loss_length", env_progress_data.loss_length_mean, episode
    )
    writer.add_scalar("episode/fps", env_progress_data.time_mean * num_envs, episode)
    # writer.add_scalar("rewards/mean_total", env_progress_data.return_mean, episode)
    writer.add_scalar("rewards/mean_winner", env_progress_data.winner_mean, episode)
    writer.add_scalar("rewards/mean_final", env_progress_data.reward_mean, episode)
    if allow_illegal:
        writer.add_scalar("rewards/mean_legal", env_progress_data.legal_mean, episode)
    # writer.add_scalar("losses/policy_loss", np.mean(policy_loss_queue), episode)
    writer.add_scalar("losses/value_loss", np.mean(value_loss_queue), episode)
    writer.add_scalar("losses/control_loss", np.mean(control_loss_queue), episode)
    writer.add_scalar("losses/legality_loss", np.mean(legality_loss_queue), episode)
    writer.add_scalar("losses/entropy", np.mean(entropy_queue), episode)
    writer.add_scalar("gradients/gnn", gnn_gradient, episode)
    writer.add_scalar("gradients/mlp", mlp_gradient, episode)
    writer.add_scalar("misc/q_mean", q_mean, episode)
    writer.add_scalar("misc/q_std", q_std, episode)
    writer.add_scalar("misc/gamma", gamma, episode)
    writer.add_scalar("misc/epsilon", eps, episode)
    writer.add_scalar("misc/buffer_priority_rate", priority_rate, episode)
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
        os.path.join(LOG_PATH, f"dqn_{board_size}", f"dqn_v{args.version}")
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
        "buffer_ratio": buffer_ratio,
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
    dqn = DQN(envs, args.color, args.version, args.load_version)
    try:
        # with th.xpu.amp.autocast(enabled=True, dtype=TH_FLOAT_TYPE):
        dqn.run(args.version)
    finally:
        writer.close()

        dqn.save_checkpoint()
