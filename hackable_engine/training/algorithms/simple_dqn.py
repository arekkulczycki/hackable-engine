import asyncio
import os
from argparse import ArgumentParser
from collections import deque
from enum import Enum
from itertools import chain
from math import exp
from random import choice
from typing import Generator

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.bitboard_utils import generate_cells
from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.common.custom_threads import ReturningTargetThread
from hackable_engine.training.algorithms.structures.chunks import Chunks
from hackable_engine.training.algorithms.util.replay_buffer import (
    ReplayBuffer,
    Experience,
    IndexedExperience,
)
from hackable_engine.training.constants import (
    LRShape,
    get_learning_rate_decay,
    TargetUpdateMode,
    GammaMode,
    BufferMode,
    LOG_PATH,
)
from hackable_engine.training.envs.multiprocess_vector_env.safe_sync_vector_env import SafeSyncVectorEnv
from hackable_engine.training.models.graph_gatformer import GraphGATformer
from hackable_engine.training.models.graph_gatgin import GraphGATGIN
from hackable_engine.training.models.graph_gengat import GraphGENGAT
from hackable_engine.training.models.graph_gmmformer import GraphGMMformer
from hackable_engine.training.utils.device import Device
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import (
    MultiprocessAsyncEnv,
)
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_gat import GraphGAT
from hackable_engine.training.models.graph_gin import GraphGIN
from hackable_engine.training.models.graph_gine import GraphGINE
from hackable_engine.training.models.graph_gmm import GraphGMM
from hackable_engine.training.models.graph_rgcn import GraphRGCN
from hackable_engine.training.models.graph_sg import GraphSG

# th._dynamo.config.cache_size_limit = 16 * 1024 * 1024 * 1024
# th._dynamo.config.suppress_errors = True
# th.set_num_threads(1)
# the following is a proposed hack for `Required aspect fp64` (raised for fused optimizer), but doesn't work
# th.Tensor.double = th.Tensor.float
# th.float64 = th.double = th.float32

class Preset(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6

preset = Preset.ZERO

entropy_by_preset = {
    Preset.ZERO: 0.1,
    Preset.ONE: 0.5,
    Preset.TWO: 0.01,
    Preset.THREE: 0.5,
    Preset.FOUR: 0.5,
    Preset.FIVE: 0.5,
    Preset.SIX: 0.5,
}
control_by_preset = {
    Preset.ZERO: 0.5,
    Preset.ONE: 0.5,
    Preset.TWO: 0.5,
    Preset.THREE: 0.5,
    Preset.FOUR: 0.1,
    Preset.FIVE: 2.0,
    Preset.SIX: 0.5,
}
a_temp_by_preset = {  # using SGD the temp can be higher, increasing exploration
    Preset.ZERO: 0.01,
    Preset.ONE:  0.01,
    Preset.TWO:  0.01,
    Preset.THREE: 0.1,
    Preset.FOUR: 0.01,
    Preset.FIVE: 0.033,
    Preset.SIX: 0.01,
}
buffer_priority_by_preset = {
    Preset.ZERO: 0.33,
    Preset.ONE: 0.33,
    Preset.TWO: 0.33,
    Preset.THREE: 0.33,
    Preset.FOUR: 0.0,
    Preset.FIVE: 0.5,  # together with big initial buffer
    Preset.SIX: 0.33,
}

# OPPONENT_VARIANTS = ["gmm1", "gmm2", "sg", "gin", "gine", "rgcn", "gat", "gatformer", "gmmformer"]
OPPONENT_VARIANTS = ["gmm110", "gmm220", "gmm330", "gmm440", "gmm550", "gmm511", "gmm522", "gmm533", "gmm544"]
# OPPONENT_VARIANTS = None
# fmt: off
GNN_SHAPES = {
    GraphGMM: (64, 96, 128, 160, 192, 224),
    GraphSG: (54, 108, 216, 324, 432),
    GraphGIN: (54, 108, 216, 432, 432, 432, 432),
    GraphGINE: (54, 108, 216, 432, 432, 432),
    GraphRGCN: (36, 54, 72, 90),
    # GraphRGCN: (54, 108, 216, 324, 432),
    GraphGAT: (36, 54, 72, 72, 72),
    GraphGENGAT: (36, 54, 72, 72, 72),
    GraphGATformer: (54, 162, 162, 162),
    GraphGMMformer: (36, 54, 54, 54),
    # GraphGMMformer: (54, 108, 108, 108, 108),
    GraphGATGIN: (54, 162, 410, 334, 256),  # grow with GAT concat=True to 486, then shrink with GIN
}
# fmt: on
model_class = GraphGMM
mlp_shape = (256,)#(gnn_shape[-1], gnn_shape[-1] // 2)  #(256,)
th_float_type = th.float32
board_size = 13
board_size_squared = board_size**2
binary_one = 2**board_size_squared - 1
env_class = Logit13GraphEnv if board_size == 13 else Logit11GraphEnv
gnn_shape = GNN_SHAPES[model_class]
gnn_heads = (3, 3, 1, 1, 1)
episode_env_steps = board_size  # **2
"""relative for tensorboard graphs, such that training sessions are comparable"""
gamma_mode = GammaMode.RETRAINED  # TODO: probably this is useless
"""if set to TRAINED then below params are ignored, use RETRAINED when model is loaded"""
num_episodes = 1024# * 2
num_envs = 128
num_workers = 8
lr_gamma = 1e-4
lr_gnn_warm_up_len = 0.33
lr_mlp_warm_up_len = 0.33
assert 0 < lr_gnn_warm_up_len <= lr_mlp_warm_up_len < 1
lr_minimum_p = 0.1
assert 0 < lr_minimum_p < 1
lr_shape_gnn = LRShape.ONE
lr_shape_mlp = LRShape.ONE
lr_gnn = 3e-4 * (1 if lr_shape_gnn is LRShape.ONE else 2)
lr_mlp = 4e-4 * (1 if lr_shape_gnn is LRShape.ONE else 2)  # should be larger than the final lr_gnn
# assert lr_mlp <= lr_gnn
sgd_momentum = 0.9
adamw_betas = (0.9, 0.999)#(0.85, 0.98)  # default: (0.9, 0.999)
weight_decay = 1e-4
"""
3e-4 likely leads to oversmoothing (underfitting), 1e-5 often a sweet spot, even 0 works.
Couple lower values with lower LR or lower TAU to avoid noisy updates.
Should be set lower for Critic than for Actor.
"""
tau_high = 0.1
tau_low = 0.005  # recommended 0.005 – 0.01 for a learning rate of 1e-3, lower for smaller learning rates
"""soft target update proportion"""
target_update_frequency = 1
target_update_mode = TargetUpdateMode.SOFT
gradient_max = None#1000.0
"""do not clip too much or learning will stagnate"""
batch_size = 256
dropouts = 0.25 if model_class not in {GraphGIN, GraphGINE} else 0.15
expected_episode_steps = 64
gamma_low = 0.1
gamma_high = (expected_episode_steps - 1) / expected_episode_steps
gamma_delay = 0.02
gae_lambda = 0.95
"""when mean rewards reach `gamma_delay` gamma becomes `gamma_high`, otherwise proportional to the rewards"""
assert not (gamma_delay <= 0.66 and gamma_mode is GammaMode.REWARD_BASED) and not (
    gamma_mode is GammaMode.ANNEALING and gamma_delay <= 0
)
epsilon_high = 0.05
epsilon_low = 0
control_weight_param_low = 0.01
control_weight_param_high = 1.0  # if model_class is not GraphGIN else 2.0
entropy_weight_param = 0.075
entropy_temperature = 2.0  # default is 1.0
"""as the entropy is incentivised by the loss function, higher value incentivises higher spread"""
action_temperature_high = 0.2
action_temperature_low = 0.01
"""
Controls how much the softmax function spreads out the probabilities over all possible actions.
A higher temp results in a wider spread of probabilities, making the agent explore more diverse actions.
Increase action temperature for exploration, decrease for exploitation.
"""
allow_illegal = True
softmax_action_selection = True
### BUFFER ###
should_load_init_buffer = False
should_dump_init_buffer = False
should_dump_final_buffer = False
buffer_priority_rate_high = 0.99 if model_class not in {GraphGIN} else 1.0
buffer_priority_rate_low = 0.33 if model_class not in {GraphGIN} else 0.5
buffer_size_low = num_envs * episode_env_steps * 32
buffer_size_high = num_envs * episode_env_steps * 2048  # keep the experience for N vector-env steps
buffer_mode = BufferMode.RAM
disk_buffer = 1 * buffer_size_high
start_training = buffer_size_low * 0.95
buffer_ratio = 8 // 2
"""Defines the proportion between fresh experience training and buffer training"""
target_model_inference_chunks = 16 if model_class is GraphGAT else 8
epochs = num_envs * buffer_ratio // batch_size
episode_step = 1
should_compile = True
should_debug = False

# action_queue = deque(maxlen=board_size**2 * num_envs)
# max_action_queue = deque(maxlen=num_envs)
# policy_loss_queue = deque(maxlen=2 * num_envs)
value_loss_queue = deque(maxlen=2 * num_envs)
control_loss_queue = deque(maxlen=2 * num_envs)
entropy_queue = deque(maxlen=2 * num_envs)


def fast_decay(episode):
    x = episode / num_episodes
    hyperbolic = 1 / (1 + 99 * x**0.5)
    # exponential = 0.995 ** episode
    # return max((hyperbolic, exponential))
    return min(1, hyperbolic)


def buffer_priority_rate_decay(episode):
    x = episode / num_episodes
    hyperbolic = 1 / (1 + 15 * x**2)
    return min(1, hyperbolic)


def sigmoid_growing(episode, lowest=0.0, highest=1.0, middle=0.5, sharpness=8):
    x = episode / num_episodes
    return (highest - lowest) / (1 + exp(-sharpness * (x - middle))) + lowest


def gamma_annealing(episode):
    x = min(1.0, episode / num_episodes / gamma_delay)
    return gamma_low + ((gamma_high - gamma_low) * (3 * x**2 - 2 * x**3))


class DQN:
    def __init__(self, envs, color: bool, version: int, load_version: int | None = None, reset_optimizer: bool = False):
        self.envs = envs
        self.envs = envs
        self.color = color
        self.version = version
        self.load_version = load_version
        model_kwargs = dict(
            node_count=board.size_square,
            node_features=9,
            output_size=1,
            batch_size=batch_size,
            dropouts=dropouts,
            num_envs=num_envs,
            gnn_shape=gnn_shape,
            # gnn_heads=gnn_heads,
            mlp_shape=mlp_shape,
            edge_index=board.edge_index,
            # edge_types=board.edge_types,
            # edge_types=board.edge_types_rgcn,
            pseudo_coordinates=board.pseudo_coordinates,
            # kernel_size=3,
            # gmm_layers=2,
            should_initialize_weights=load_version is None
        )
        self.model = (
            model_class(**model_kwargs)
            .to(Device.XPU)
            .to(th_float_type)
        )
        self.target_model = (
            model_class(**{**model_kwargs, "dropouts": 0.0})
            .to(Device.XPU)
            .to(th_float_type)
        )
        # will switch to train() when replay buffer is loaded
        self.model.eval()
        self.target_model.eval()

        if load_version is not None:
            color_name = "white" if self.color else "black"
            model_weights = th.load(f"dqn/{board_size}/dqn-model-{color_name}.v{load_version}", map_location="xpu")
            model_weights = {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}
            self.model.load_state_dict(model_weights, strict=True, assign=True)
            target_weights = th.load(
                f"dqn/{board_size}/dqn-target-model-{color_name}.v{load_version}", map_location="xpu"
            )
            target_weights = {k.replace("_orig_mod.", ""): v for k, v in target_weights.items()}
            self.target_model.load_state_dict(target_weights, strict=True, assign=True)
            print(f"weights from color {color_name} version {load_version} loaded")
        else:
            self.target_model.load_state_dict(self.model.state_dict())

        self.tmp_value_loss = th.tensor(0, dtype=th_float_type, device=Device.XPU)
        self.tmp_control_loss = th.tensor(0, dtype=th_float_type, device=Device.XPU)
        self.tmp_entropy = th.tensor(0, dtype=th_float_type, device=Device.XPU)

        optimizer_params = [
            {
                # "params": [param for layer in chain(self.model.gnn, self.model.residuals, self.model.norms) for param in layer.parameters()],
                "params": [param for layer in chain(self.model.gnn, self.model.residuals) for param in layer.parameters()],
                "lr": lr_gnn,
            },
            {
                "params": [param for layer in self.model.mlp for param in layer.parameters()],
                "lr": lr_mlp,
            },
            {
                "params": [param for layer in self.model.control_mlp for param in layer.parameters()],
                "lr": lr_mlp,
            },
        ]
        scheduler_params = [
            get_learning_rate_decay(lr_shape_gnn, num_episodes, lr_gnn_warm_up_len, lr_minimum_p),
            get_learning_rate_decay(lr_shape_mlp, num_episodes, lr_mlp_warm_up_len, lr_minimum_p),
            get_learning_rate_decay(lr_shape_mlp, num_episodes, lr_mlp_warm_up_len, lr_minimum_p),
        ]
        if gamma_mode is GammaMode.TRAINED:
            self.model.gamma = nn.Parameter(th.tensor(-2, dtype=th_float_type, device=Device.XPU))
            self.target_model.gamma = nn.Parameter(th.tensor(-2, dtype=th_float_type, device=Device.XPU))
            optimizer_params.append({"params": self.model.gamma, "lr": lr_gamma})
            scheduler_params.append(
                get_learning_rate_decay(LRShape.ONE, num_episodes, lr_mlp_warm_up_len, lr_minimum_p)
            )

        # self.optimizer = optim.Adadelta(optimizer_params, weight_decay=weight_decay)
        self.optimizer = optim.SGD(optimizer_params, weight_decay=weight_decay, momentum=sgd_momentum, nesterov=True)
        # self.optimizer = optim.AdamW(
        #     optimizer_params,
        #     weight_decay=weight_decay,
        #     betas=adamw_betas,
        #     # fused=True,
        # )
        self.replay_buffer = ReplayBuffer(
            board_size=board_size,
            capacity=buffer_size_low,
            storage_type=ReplayBuffer.StorageType.LIST,
            priority_rate=buffer_priority_rate_high,
        )
        self.optimizer_scheduler = LambdaLR(
            self.optimizer,
            scheduler_params,
        )

        if load_version is not None and not reset_optimizer:
            color_name = "white" if self.color else "black"
            optimizer_weights = th.load(
                f"dqn/{board_size}/dqn-optimizer-{color_name}.v{load_version}", map_location="xpu"
            )
            self.optimizer.load_state_dict(optimizer_weights)

        # th.set_float32_matmul_precision("medium")  # this currently only affects Nvidia GPUs
        th.set_float32_matmul_precision("high")
        # print(th._inductor.list_mode_options())
        if should_compile:
            th._dynamo.reset()
            self.model = th.compile(self.model)
            # self.model = th.compile(self.model, mode="max-autotune-no-cudagraphs")
            # self.model = th.compile(self.model, backend="inductor", mode="reduce-overhead")
            # self.model = th.compile(self.model, backend="openxla") # or openvino
            self.target_model = th.compile(self.target_model)
            # self.target_model = th.compile(self.target_model, mode="max-autotune-no-cudagraphs")
            # self.target_model = th.compile(self.target_model, backend="inductor", mode="reduce-overhead")
            # self.model = th.compile(self.model, backend="openxla") # or openvino
            self.run_target_model = th.compile(self.run_target_model)
            self.train_in_epochs = th.compile(self.train_in_epochs)
            # self.calculate_loss = th.compile(self.calculate_loss)
            # self.loss_backward = th.compile(self.loss_backward)

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
        eps = 0
        tau = tau_high
        e_temperature = entropy_temperature
        a_temperature = action_temperature_high
        control_weight_param = control_weight_param_low
        # gamma = gamma_low
        target_update_f = target_update_frequency // 2 if target_update_frequency >= 8 else target_update_frequency

        if not allow_illegal:
            self.replay_buffer.capacity = buffer_size_high

        self.obs: list[np.ndarray]
        self.black_masks: tuple[int, ...]
        self.white_masks: tuple[int, ...]
        self.obs, self.black_masks, self.white_masks = self.envs.reset()
        old_experiences: Generator[IndexedExperience] = (x for x in ())
        new_experiences: list[IndexedExperience] = []

        ep_start = 0
        for i in range(ep_start):
            self.optimizer_scheduler.step()

        for episode in range(ep_start, num_episodes, episode_step):  # TODO: start from 0 again
            print("episode", episode, "buffer", self.replay_buffer.size())
            steps = 0
            while True:
                steps += 1

                # th_obs = th.from_numpy(np.stack(self.obs, 0)).to(Device.XPU)
                # q_values_chunks = (self.model(chunk) for chunk in th_obs.chunk(16))
                # q_values = th.cat([chunk[0] for chunk in q_values_chunks])
                gpu_obs = th.from_numpy(np.stack(self.obs, 0)).to(Device.XPU)
                if self.training_started:
                    with th.amp.autocast(Device.XPU, dtype=th.bfloat16, enabled=False):
                        q_values, _ = self.model(gpu_obs)  # in training mode includes control head output
                    actions = self.get_logit_actions(eps, a_temperature, q_values, allow_illegal)
                else:
                    with th.amp.autocast(Device.XPU, dtype=th.bfloat16, enabled=False):
                        q_values = self.model(gpu_obs)
                    actions = self.get_logit_actions(eps, a_temperature, q_values, args.load_version is not None)

                # action_queue.extend(actions.flatten().tolist())

                # t0 = perf_counter()
                # self.obs, self.illegal_masks, new_experiences = await self.run_env(self.obs, actions)
                # print("env takes", perf_counter() - t0)
                #
                # t0 = perf_counter()
                # value_loss, entropy, gamma, old_experiences = await self.train_dqn_on_buffer(gamma, new_experiences, old_experiences)
                # print("training takes", perf_counter() - t0)
                (self.obs, self.black_masks, self.white_masks, new_experiences), (
                    value_loss,
                    control_loss,
                    entropy,
                    old_experiences,
                ) = await asyncio.gather(
                    self.run_env(self.obs, actions),
                    self.train_dqn_on_buffer(new_experiences, old_experiences, e_temperature, control_weight_param),
                )

                if value_loss and steps > episode_env_steps:
                    value_loss_queue.append(value_loss)
                    control_loss_queue.append(control_loss)
                    entropy_queue.append(entropy)

                    break

            if self.replay_buffer.size() < start_training:
                continue

            ### DECAYING AND ANNEALING PARAMETERS ###
            decay = fast_decay(episode)
            eps = (epsilon_high - epsilon_low) * decay + epsilon_low if args.load_version is None else 0.0
            a_temperature = (action_temperature_high - action_temperature_low) * decay + action_temperature_low
            tau = tau_low if args.load_version is not None else (tau_high - tau_low) * decay + tau_low
            control_weight_param = sigmoid_growing(episode, control_weight_param_low, control_weight_param_high, middle=0.2)
            self.replay_buffer.priority_rate = buffer_priority_rate_low if args.load_version else ((
                buffer_priority_rate_high - buffer_priority_rate_low
            ) * buffer_priority_rate_decay(episode) + buffer_priority_rate_low)
            # if gamma_mode is GammaMode.ANNEALING:
            #     gamma = gamma_low + ((gamma_high - gamma_low) * gamma_annealing(episode))

            for i in range(episode_step):
                self.optimizer_scheduler.step()

            remainder = 1 if target_update_frequency == 2 else 2
            if target_update_frequency == 1 or episode % target_update_f == remainder:
                print("updating target")
                self.update_target(tau)
                if target_update_f != target_update_frequency and episode > target_update_f:
                    target_update_f = target_update_frequency
                    # if gamma > 2 * gamma_low:
                    self.replay_buffer.capacity = buffer_size_high
                    if buffer_mode is BufferMode.RAM_AND_DISK:
                        self.replay_buffer.backup.active = True

            if episode % 1 == 0:
                q_mean = q_values.mean().item()
                q_std = q_values.std().item()

                gnn_gradients = []
                for param in [param for layer in self.model.gnn for param in layer.parameters()]:
                    if param.grad is not None:
                        gnn_gradients.append(param.grad.view(-1))
                gnn_gradients = th.cat(gnn_gradients)
                gnn_gradient = gnn_gradients.norm()
                del gnn_gradients

                res_gradients = []
                # for param in [param for layer in chain(self.model.residuals, self.model.norms) for param in layer.parameters()]:
                for param in [param for layer in self.model.residuals for param in layer.parameters()]:
                    if param.grad is not None:
                        res_gradients.append(param.grad.view(-1))
                res_gradients = th.cat(res_gradients)
                res_gradient = res_gradients.norm()
                del res_gradients

                mlp_gradients = []
                for param in [param for layer in self.model.mlp for param in layer.parameters()]:
                    if param.grad is not None:
                        mlp_gradients.append(param.grad.view(-1))
                mlp_gradients = th.cat(mlp_gradients)
                mlp_gradient = mlp_gradients.norm()
                del mlp_gradients

                lr_gnn_, lr_mlp_ = self.optimizer_scheduler.get_last_lr()[:2]
                self.log_progress(
                    episode,
                    q_mean,
                    q_std,
                    gnn_gradient,
                    res_gradient,
                    mlp_gradient,
                    eps,
                    tau,
                    a_temperature,
                    control_weight_param,
                    lr_gnn_,
                    lr_mlp_,
                    self.replay_buffer.priority_rate,
                )
                # log_progress(episode, gradient, eps, gamma.item(), lr_gnn_, lr_mlp_)
            if episode % 64 == 0 and episode > 0:
                self.save_checkpoint(episode)

    async def run_env(
        self, last_obss, actions
    ) -> tuple[np.array, tuple[int, ...], tuple[int, ...], list[IndexedExperience]]:
        obss, rewards, dones, _, black_masks, white_masks = await envs.step(actions)
        for ndarr in obss:
            if np.isnan(ndarr).any():
                raise ValueError("nan in fucking obss")

        experiences: list[IndexedExperience] = []
        buffer_indices = np.random.randint(0, self.replay_buffer.capacity, num_envs)
        for (
            buffer_index,
            last_obs,
            action,
            reward,
            obs,
            done,
            black_mask,
            white_mask,
        ) in zip(
            buffer_indices,
            last_obss,
            actions,
            rewards,
            obss,
            dones,
            black_masks,
            white_masks,
        ):
            experiences.append((last_obs, action, reward, obs, done, black_mask, white_mask, buffer_index))
            self.replay_buffer.push(buffer_index, (last_obs, action, reward, obs, done, black_mask, white_mask))

        return obss, black_masks, white_masks, experiences

    async def train_dqn_on_buffer(
        self,
        new_experiences: list[IndexedExperience],
        old_experiences: Generator[IndexedExperience, None, None],
        e_temperature: float,
        control_weight_param: float,
    ) -> tuple[float, float, float, tuple[Experience, ...]]:
        prev_value_loss: float = 0
        prev_control_loss: float = 0
        prev_entropy: float = 0
        size = self.replay_buffer.size()
        if size > start_training:
            if not self.training_started:
                self.training_started = True
                self.model.train()
                print("started training")
                if should_dump_init_buffer:
                    await self.replay_buffer.dump_to_disk(self.envs.single_observation_space.shape)
                    print("dumped initial buffer to disk")

            if not old_experiences:
                old_experiences = tuple(self.replay_buffer.sample(num_envs * (buffer_ratio - 1)))

            states, actions, rewards, next_states, dones, black_masks, white_masks, buffer_indices = zip(
                *chain(new_experiences, old_experiences)
            )

            next_states = th.from_numpy(np.array(next_states, dtype=FLOAT_TYPE))
            next_states = next_states.pin_memory().to(device=Device.XPU, non_blocking=True)

            thread = ReturningTargetThread(
                target=self.run_target_model,
                args=(
                    self.target_model,
                    next_states,
                ),
            )
            thread.start()

            control_stats_list = []
            for black_mask, white_mask in zip(black_masks, white_masks):
                control_stats_list.append(self.replay_buffer.find_closest_prob_oc_stats(black_mask, white_mask))

            control_stats_stacked = th.from_numpy(np.stack(control_stats_list))

            next_experiences = self.replay_buffer.sample(num_envs * (buffer_ratio - 1))

            states = th.from_numpy(np.array(states, dtype=FLOAT_TYPE))
            actions = th.from_numpy(np.array(actions, dtype=np.int64))
            rewards = th.from_numpy(np.array(rewards, dtype=FLOAT_TYPE))
            dones = th.from_numpy(np.array(dones, dtype=FLOAT_TYPE))

            # gamma = self.calculate_gamma(rewards, dones, last_gamma)

            states = states.pin_memory().to(device=Device.XPU, non_blocking=True)
            actions = actions.pin_memory().to(device=Device.XPU, non_blocking=True)
            rewards = rewards.pin_memory().to(device=Device.XPU, non_blocking=True)
            dones = dones.pin_memory().to(device=Device.XPU, non_blocking=True)
            control_stats_stacked = control_stats_stacked.pin_memory().to(device=Device.XPU, non_blocking=True)

            buffer_indices_chunks = np.array_split(np.array(buffer_indices), epochs, axis=0)
            states_chunks = states.chunk(epochs, dim=0)
            next_states_chunks = next_states.chunk(epochs, dim=0)
            actions_chunks = actions.chunk(epochs, dim=0)
            rewards_chunks = rewards.chunk(epochs, dim=0)
            dones_chunks = dones.chunk(epochs, dim=0)
            control_stats_chunks = control_stats_stacked.chunk(epochs, dim=0)

            # copying loss value from gpu takes significant time so let's do it while target thread is running
            prev_value_loss = self.tmp_value_loss.item() / epochs
            prev_control_loss = self.tmp_control_loss.item() / epochs
            prev_entropy = self.tmp_entropy.item() / epochs
            control_loss_weight = abs(prev_value_loss / prev_control_loss) if prev_control_loss else 1.0
            entropy_weight = -abs(prev_value_loss / prev_entropy) if prev_entropy else -1.0
            self.tmp_value_loss.zero_()
            self.tmp_control_loss.zero_()
            self.tmp_entropy.zero_()

            target_q_values = thread.join()
            # target_q_values = target_q_values.detach()

            # target_q_values = self.target_model(next_states).detach()
            target_q_values_chunks = target_q_values.chunk(epochs, dim=0)

            self.train_in_epochs(
                Chunks(
                    buffer_indices_chunks,
                    states_chunks,
                    next_states_chunks,
                    actions_chunks,
                    rewards_chunks,
                    dones_chunks,
                    control_stats_chunks,
                    target_q_values_chunks,
                ),
                e_temperature,
                control_loss_weight,
                entropy_weight,
                control_weight_param,
            )
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
                await self.replay_buffer.load_init_buffer(self.envs.single_observation_space.shape, buffer_size_low)
                size = self.replay_buffer.size()
                print("initial buffer loaded")
            if size >= num_envs * (buffer_ratio - 1):
                next_experiences = tuple(self.replay_buffer.sample(num_envs * (buffer_ratio - 1)))
            else:
                next_experiences = ()
            print(f"{np.round(size / start_training * 100, 2)} %")

        return (
            prev_value_loss,
            prev_control_loss,
            prev_entropy,
            next_experiences,
        )

    @staticmethod
    # @th.compile(disable=False)
    def run_target_model(target_model, states):
        with th.no_grad(), th.amp.autocast(Device.XPU, dtype=th.bfloat16, enabled=False):
            try:
                chunks = states.chunk(target_model_inference_chunks)
                # print("calculating target model...")
                # t0 = perf_counter()
                # with th.profiler.profile(activities=[th.profiler.ProfilerActivity.XPU]) as prof:
                q_values_chunks = [target_model(chunk).detach() for chunk in chunks]
                # print(prof.key_averages().table(sort_by="xpu_time_total", row_limit=20))
                # print("target model calculated in", perf_counter() - t0)
                return th.cat(q_values_chunks)
            except Exception as e:
                print("*************\n", e)

    # async def train_ppo_on_buffer(self, states, next_states, rewards, dones, actions):
    #     # Compute value of current states (V(s))
    #     # state_values = critic(states).squeeze()
    #     # action_probs = actor(states)
    #     state_values, action_probs = self.model(states)
    #
    #     # Compute value of next states (V(s'))
    #     # next_state_values = critic(next_states).squeeze()
    #     next_state_values = self.model.forward_value(next_states)
    #
    #     # Compute TD error (delta_t)
    #     deltas = rewards + gamma_high * next_state_values * (1 - dones) - state_values
    #
    #     # Compute the GAE (Generalized Advantage Estimation)
    #     advantages = th.zeros_like(deltas)
    #     running_advantage = 0
    #     for t in reversed(range(len(deltas))):
    #         running_advantage = deltas[t] + gamma_high * gae_lambda * running_advantage * (1 - dones[t])
    #         advantages[t] = running_advantage
    #
    #     td_target = rewards + gamma_high * next_state_values * (1 - dones)
    #
    #     # Critic Loss (Mean Squared Error)
    #     critic_loss = F.mse_loss(state_values, td_target)
    #
    #     # Actor Loss (Policy Gradient Loss)
    #     action_log_probs = th.log(action_probs)
    #     selected_action_log_probs = action_log_probs.gather(
    #         1, actions.unsqueeze(1)
    #     )  # Select log-prob for taken actions
    #     actor_loss = -th.mean(selected_action_log_probs * advantages)  # Negative because we do gradient ascent
    #
    #     # Total Loss = Actor Loss + Critic Loss
    #     total_loss = actor_loss + critic_loss
    #
    #     # Optimize the actor and critic networks
    #     # actor_optimizer.zero_grad()
    #     # critic_optimizer.zero_grad()
    #     self.optimizer.zero_grad()
    #
    #     total_loss.backward()
    #
    #     # actor_optimizer.step()
    #     # critic_optimizer.step()
    #     self.optimizer.step()

    def get_logit_actions(self, eps: float, a_temperature: float, q_values, allow_illegal_: bool):
        """The action chosen is the highest value regardless of color, using `argmax`."""

        if allow_illegal_:
            return np.array(
                [
                    (
                        choice(list(generate_cells(binary_one ^ (self.black_masks[i] | self.white_masks[i]))))
                        if np.random.rand() < eps
                        else (
                            self.get_softmax_action(q_values[i], a_temperature)
                            if softmax_action_selection
                            else q_values[i].argmax().item()
                        )
                    )
                    for i in range(num_envs)
                ],
                dtype=FLOAT_TYPE,
            )
        else:
            # FIXME: the following produces illegal actions
            # return np.array(
            #     [
            #         (
            #             choice(list(generate_cells(binary_one ^ (self.black_masks[i] | self.white_masks[i]))))
            #             if np.random.rand() < eps
            #             else q_values[i][
            #                 list(generate_cells(binary_one ^ (self.black_masks[i] | self.white_masks[i])))
            #             ].argmax().item()
            #         )
            #         for i in range(num_envs)
            #     ],
            #     dtype=FLOAT_TYPE,
            # )
            # fmt: on
            actions = []
            for i, (black_mask, white_mask) in enumerate(zip(self.black_masks, self.white_masks)):
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
            # fmt: on
            return np.array(actions, dtype=FLOAT_TYPE)

    @staticmethod
    def get_softmax_action(q_values, a_temperature: float) -> int:
        probabilities = F.softmax(q_values / a_temperature, dim=-1)
        return th.multinomial(probabilities, num_samples=1).item()

    @staticmethod
    def get_logit_entropy(q_values):
        probs = F.softmax(q_values.clone().detach(), dim=-1)
        return -th.sum(probs * (probs + 1e-8).log(), dim=-1).mean()

    def train_in_epochs(self, chunks: Chunks, e_temperature: float, control_loss_weight: float, entropy_weight: float, control_weight_param: float):
        for i in range(epochs):
            buffer_indices_ = chunks.buffer_indices_chunks[i]
            states_ = chunks.states_chunks[i]
            next_states_ = chunks.next_states_chunks[i]
            actions_ = chunks.actions_chunks[i]
            rewards_ = chunks.rewards_chunks[i]
            target_q_values_ = chunks.target_q_values_chunks[i]
            dones_ = chunks.dones_chunks[i]
            control_stats = chunks.control_stats_chunks[i]

            # scaler = th.amp.GradScaler(Device.XPU)
            with th.autograd.set_detect_anomaly(should_debug):
                value_loss, control_loss, entropy = self.get_logit_loss(
                    states_,
                    next_states_,
                    target_q_values_,
                    rewards_,
                    dones_,
                    actions_,
                    control_stats,
                    e_temperature,
                    buffer_indices_,
                )
                del states_, actions_, rewards_, target_q_values_, dones_

                self.optimizer.zero_grad()
                # self.loss_backward(
                #     value_loss, control_loss, entropy, control_loss_weight, entropy_weight, control_weight_param
                # )
                (
                    value_loss
                    + control_loss * control_loss_weight * control_weight_param
                    + entropy * entropy_weight * entropy_weight_param
                ).backward()

                if gradient_max:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_max)

                self.optimizer.step()

            self.tmp_value_loss += value_loss.detach()
            self.tmp_control_loss += control_loss.detach()
            self.tmp_entropy += entropy.detach()

    # @th.compile(disable=False)
    def get_logit_loss(
        self,
        states,
        next_states,
        target_q_values,
        rewards,
        dones,
        actions,
        control_stats: th.Tensor,
        e_temperature: float,
        buffer_indices: np.ndarray,
    ):
        value_loss, control_loss, entropy, q_predicted, target = self.calculate_loss(states, next_states, actions, target_q_values, rewards, dones, control_stats, e_temperature)

        # has_error = False
        # # fmt: off
        # for i, tensr in enumerate([value_loss,batched,q_values_all,control_all,q_values,next_q_online,control,best_actions,target_q,target]):
        #     if th.isnan(tensr).any():
        #         has_error = True
        #         print(f"nan in tensor {i}")
        # # fmt: on        # if has_error:        #     for name, param in self.model.named_parameters():
        #         if th.isnan(param).any():
        #             print(f"NaNs in weights: {name}")
        #     raise ValueError("nan in tensor")

        # set prioritization in the buffer
        td_errors = th.abs(q_predicted - target).detach()
        cpu_errors = td_errors.cpu()
        np_errors = cpu_errors.numpy()
        self.replay_buffer.td_errors[buffer_indices] = np_errors

        return (
            value_loss,
            control_loss,
            entropy,
        )

    def calculate_loss(self, states, next_states, actions, target_q_values, rewards, dones, control_stats, e_temperature):
        # VANILLA DQN
        # q_values, control = self.model(states)
        # target = rewards + gamma * target_q_values.max(dim=1)[0] * (1 - dones)

        # DOUBLE DQN
        batched = th.cat([states, next_states], dim=0)
        q_values_all, control_all = self.model(batched)

        q_values, next_q_online = q_values_all.chunk(2, dim=0)
        control, _ = control_all.chunk(2, dim=0)
        best_actions = next_q_online.argmax(dim=1)
        q_predicted = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        target_q = target_q_values.gather(1, best_actions.unsqueeze(1)).squeeze(1)
        target = rewards + gamma_high * target_q * (1 - dones)

        # TODO: monitor td_error and the following, periodically if heavy
        # q_drift = F.mse_loss(q_values, target_q_values, reduction='mean').item()
        # control_log_probs = (F.softmax(control, dim=-1) + 1e-8).log()
        control_log_probs = F.log_softmax(control, dim=-1)
        # control_loss = -(control_log_probs * control_stats).mean()
        control_loss = F.kl_div(control_log_probs, control_stats, reduction="batchmean")
        q_probs = F.softmax(q_values / e_temperature, dim=-1)
        entropy = -(q_probs * (q_probs + 1e-10).log()).sum(dim=-1).mean()
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss_function = F.mse_loss  # F.smooth_l1_loss if model_class is GraphGIN else F.mse_loss
        value_loss = loss_function(q_value, target)

        return value_loss, control_loss, entropy, q_predicted, target

    @staticmethod
    # @th.compile(disable=True)
    def loss_backward(value_loss, control_loss, entropy, control_loss_weight, entropy_weight, control_weight_param):
        (
            value_loss
            + control_loss * control_loss_weight * control_weight_param
            + entropy * entropy_weight * entropy_weight_param
        ).backward()

    @staticmethod
    def huber_loss(q_value: th.Tensor, target: th.Tensor, delta: int = 1.0):
        """Alternative to F.mse_loss, custom version of F.smooth_l1_loss."""
        error = q_value - target
        abs_error = th.abs(error)
        quadratic = th.minimum(abs_error, th.tensor(delta))
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + delta * linear
        return loss.mean()

    def update_target(self, tau):
        if target_update_mode is TargetUpdateMode.HARD:
            self.target_model.load_state_dict(self.model.state_dict())
        elif target_update_mode is TargetUpdateMode.SOFT:
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
        if gamma_mode is GammaMode.REWARD_BASED:
            sum_reward = (rewards * dones).sum().item()
            if sum_reward == 0:
                return last_gamma

            # WARNING: rewards are not from recent experiences, but from replay sample
            mean_reward = sum_reward / dones.sum().item()
            reward_curve_point = -(mean_reward**2) / 3 + mean_reward / 2 + 1 / 3 if mean_reward < 0.75 else 0.66
            a = min(max(gamma_delay - reward_curve_point, 0), 1)
            """starts from 1, gets closer to 0 as rewards grow, ==0 from reward>0.5, >0.5 from reward>-0.5"""
            gamma = gamma_low * a + gamma_high * (1 - a)
            # print("mean rew", mean_reward)
            return (gamma + last_gamma) / 2
        elif gamma_mode is GammaMode.ANNEALING:
            return last_gamma
        elif gamma_mode is GammaMode.TRAINED:
            return F.sigmoid(self.model.gamma)
        elif gamma_mode is GammaMode.RETRAINED:
            return gamma_high

        raise ValueError("unknown gamma mode")

    def save_checkpoint(self, episode: int | None = None):
        color_name = "white" if self.color else "black"
        version = self.version if episode is None else f"{self.version}.{episode}"

        if not os.path.exists(f"dqn/{board_size}"):
            os.mkdir(f"dqn/{board_size}")

        if should_compile:
            th.save(self.model._orig_mod.state_dict(), f"dqn/{board_size}/dqn-model-{color_name}.v{version}")
            th.save(
                self.target_model._orig_mod.state_dict(),
                f"dqn/{board_size}/dqn-target-model-{color_name}.v{version}",
            )
        else:
            th.save(self.model.state_dict(), f"dqn/{board_size}/dqn-model-{color_name}.v{version}")
            th.save(
                self.target_model.state_dict(),
                f"dqn/{board_size}/dqn-target-model-{color_name}.v{version}",
            )
        th.save(
            self.optimizer.state_dict(),
            f"dqn/{board_size}/dqn-optimizer-{color_name}.v{version}",
        )

    def log_progress(
        self,
        episode,
        q_mean,
        q_std,
        gnn_gradient,
        res_gradient,
        mlp_gradient,
        eps,
        tau,
        temperature,
        control_weight_param,
        lr_gnn_,
        lr_mlp_,
        priority_rate,
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
        if env_progress_data.reward_mean > -0.5:
            self.replay_buffer.should_sample_illegal = False

        writer.add_scalar("episode/length", env_progress_data.length_mean, episode)
        writer.add_scalar("episode/win_length", env_progress_data.win_length_mean, episode)
        writer.add_scalar("episode/win_length_std", env_progress_data.win_length_std, episode)
        writer.add_scalar("episode/loss_length", env_progress_data.loss_length_mean, episode)
        writer.add_scalar("episode/loss_length_std", env_progress_data.loss_length_std, episode)
        writer.add_scalar("episode/fps", env_progress_data.time_mean * num_envs, episode)
        writer.add_scalar("rewards/mean_total", env_progress_data.return_mean, episode)
        writer.add_scalar("rewards/mean_winner", env_progress_data.winner_mean, episode)
        writer.add_scalar("rewards/mean_final", env_progress_data.reward_mean, episode)
        if allow_illegal:
            # % of games that finished without any illegal moves
            writer.add_scalar("rewards/%_legal", env_progress_data.legal_mean, episode)
        # writer.add_scalar("losses/policy_loss", np.mean(policy_loss_queue), episode)
        writer.add_scalar("losses/value_loss", np.mean(value_loss_queue), episode)
        writer.add_scalar("losses/control_loss", np.mean(control_loss_queue), episode)
        writer.add_scalar("losses/entropy", np.mean(entropy_queue), episode)
        writer.add_scalar("gradients/gnn", gnn_gradient, episode)
        writer.add_scalar("gradients/res", res_gradient, episode)
        writer.add_scalar("gradients/mlp", mlp_gradient, episode)
        writer.add_scalar("misc/q_mean", q_mean, episode)
        writer.add_scalar("misc/q_std", q_std, episode)
        writer.add_scalar("misc/control_weight", control_weight_param, episode)
        # writer.add_scalar("misc/gamma", gamma, episode)
        writer.add_scalar("misc/epsilon", eps, episode)
        writer.add_scalar("misc/tau", tau, episode)
        writer.add_scalar("misc/temperature", temperature, episode)
        writer.add_scalar("misc/buffer_priority_rate", priority_rate, episode)
        writer.add_scalar("misc/lr_gnn", lr_gnn_, episode)
        writer.add_scalar("misc/lr_mlp", lr_mlp_, episode)


def to_probs(actions):
    return F.sigmoid(actions)
    # mean = actions.mean()
    # std = actions.std()
    # return (F.tanh((actions - mean)/(std + 1e-8)) + 1) / 2


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
        "-r",
        "--reset-optimizer",
        action="store_true",
        help="if should reset optimizer weights, only applies when loading version for retraining",
    )
    parser.add_argument("-v", "--version", type=int, help="version of the model to save", required=True)
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

    board = HexBoard("", size=board_size)
    writer = SummaryWriter(os.path.join(LOG_PATH, f"dqn_{board_size}_{args.color}", f"{model_class.__name__.lower()}-{args.color}-v{args.version}"))
    params = {
        "num_envs": num_envs,
        "num_workers": num_workers,
        "num_episodes": num_episodes,
        "lr_gnn": lr_gnn,
        "lr_mlp": lr_mlp,
        "lr_shape_gnn": lr_shape_gnn,
        "lr_shape_mlp": lr_shape_mlp,
        "lr_gnn_warm_up_len": lr_gnn_warm_up_len,
        "lr_mlp_warm_up_len": lr_mlp_warm_up_len,
        "batch_size": batch_size,
        "buffer_ratio": buffer_ratio,
        "target_update_freq": target_update_frequency,
        "target_update_mode": target_update_mode,
        "tau_high": tau_high,
        "tau_low": tau_low,
        "gamma_mode": gamma_mode,
        "gamma_low": gamma_low,
        "gamma_high": gamma_high,
        "gamma_delay": gamma_delay,
        "buffer_size_low": buffer_size_low,
        "buffer_size_high": buffer_size_high,
    }
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in params.items()])),
        global_step=0,
    )

    # envs = MultiprocessAsyncEnv(
    #     lambda process_id, num_envs, color_: EpisodeStats(
    #         BatchInferenceVectorEnv(
    #             [
    #                 lambda: env_class(
    #                     color=color_,
    #                     process_id=process_id,
    #                     env_id=env_id,
    #                     num_processes=num_workers,
    #                     num_envs=num_envs,
    #                 )
    #                 for env_id in range(num_envs)
    #             ],
    #             copy=False,
    #             color=color_,
    #             model_variants=OPPONENT_VARIANTS,
    #         ),
    #         is_multiprocessed=True,
    #     ),
    #     num_workers,
    #     int(num_envs // num_workers),
    #     action_shape=(1,),
    #     color=args.color,
    # )
    envs = MultiprocessAsyncEnv(
        lambda process_id, num_envs, color_: EpisodeStats(
            SafeSyncVectorEnv(
                [
                    lambda: env_class(
                        color=color_,
                        process_id=process_id,
                        env_id=env_id,
                        num_processes=num_workers,
                        num_envs=num_envs,
                        models=OPPONENT_VARIANTS,
                    )
                    for env_id in range(num_envs)
                ],
                copy=False,
            ),
            is_multiprocessed=True,
        ),
        num_workers,
        int(num_envs // num_workers),
        action_shape=(1,),
        color=args.color,
    )
    dqn = DQN(envs, args.color, args.version, args.load_version, args.reset_optimizer)
    try:
        # with th.xpu.amp.autocast(enabled=True, dtype=TH_FLOAT_TYPE):
        dqn.run(args.version)
    finally:
        writer.close()

        dqn.save_checkpoint()
        if should_dump_final_buffer:
            print("dumping final buffer...")
            asyncio.run(
                dqn.replay_buffer.dump_to_disk(
                    dqn.envs.single_observation_space.shape,
                    "/var/tmp/hackable_engine/final_buffer",
                )
            )
            print("dumped final buffer")
