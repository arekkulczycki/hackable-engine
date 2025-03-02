# -*- coding: utf-8 -*-
import os
import random
from contextlib import nullcontext

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import intel_extension_for_pytorch as ipex
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    SAGEConv,
    GCNConv,
    ResGatedGraphConv,
    TransformerConv,
    EGConv,
)

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import TH_FLOAT_TYPE
from hackable_engine.training.hyperparams import *

LOG_PATH = "./hackable_engine/training/logs/"


class QNetwork(nn.Module):
    def __init__(self, env, mlp_size, edge_index):
        super().__init__()

        self.edge_index = edge_index
        self.batch_edge_index = None
        # https://github.com/reshalfahsi/node-classification/blob/master/Node_Classification.ipynb https://github.com/reshalfahsi/node-classification?tab=readme-ov-file
        # self.resgated = ResGatedGraphConv(1, 6)
        # self.sage = SAGEConv(6, 2 * 6)
        # self.transformer = TransformerConv(2 * 6, 2 * 6)
        self.conv1 = SAGEConv(1, 6)
        # # self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(6, 6)
        # self.prelu2 = nn.PReLU()
        self.conv3 = SAGEConv(6, 12)
        # for module in [self.conv1, self.conv2]:
        #     # th.nn.init.xavier_normal_(module.lin.weight)
        #     th.nn.init.kaiming_normal_(
        #         module.lin.weight, mode="fan_out", nonlinearity="relu"
        #     )
        #     th.nn.init.zeros_(module.bias)

        self.fc1 = nn.Linear(
            # np.array(env.single_observation_space.shape).prod()
            # + np.prod(env.single_action_space.shape),
            12 * 49 + 1,
            mlp_size,
        )
        self.fc2 = nn.Linear(mlp_size, mlp_size)
        self.fc3 = nn.Linear(mlp_size, 1)

    def forward(self, x, a):
        if x.shape[0] != N_ENVS:
            edge_index = _get_batch_edge_index(x, self.edge_index)
        elif self.batch_edge_index is None:
            self.batch_edge_index = _get_batch_edge_index(x, self.edge_index)
            edge_index = self.batch_edge_index
        else:
            edge_index = self.batch_edge_index

        # # x = self.prelu1(self.conv1(x, self.edge_index))
        # # x = self.prelu2(self.conv2(x, self.edge_index))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # x = self.conv2(x, edge_index)
        x = x.reshape(x.shape[0], 12 * 49)

        # x = x.flatten(1, -1)
        x = th.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env, mlp_size, edge_index):
        super().__init__()

        self.edge_index = edge_index
        self.batch_size = N_ENVS
        self.batch_edge_index = edge_index
        # https://github.com/reshalfahsi/node-classification/blob/master/Node_Classification.ipynb https://github.com/reshalfahsi/node-classification?tab=readme-ov-file
        # self.resgated = ResGatedGraphConv(1, 6)
        # self.sage = SAGEConv(6, 2 * 6)
        # self.transformer = TransformerConv(2 * 6, 2 * 6)
        self.conv1 = SAGEConv(1, 6)
        # self.prelu1 = nn.PReLU()
        self.conv2 = SAGEConv(6, 6)
        # self.prelu2 = nn.PReLU()
        self.conv3 = SAGEConv(6, 12)
        # for module in [self.conv1, self.conv2]:
        #     # th.nn.init.xavier_normal_(module.lin.weight)
        #     th.nn.init.kaiming_normal_(
        #         module.lin.weight, mode="fan_out", nonlinearity="relu"
        #     )
        #     th.nn.init.zeros_(module.bias)

        self.fc1 = nn.Linear(
            # np.array(env.single_observation_space.shape).prod(),
            12 * 49,
            mlp_size,
        )
        self.fc2 = nn.Linear(mlp_size, mlp_size)
        self.fc_mu = nn.Linear(mlp_size, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            th.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=TH_FLOAT_TYPE,
            ),
        )
        self.register_buffer(
            "action_bias",
            th.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=TH_FLOAT_TYPE,
            ),
        )

    def forward(self, x):
        if x.shape[0] != N_ENVS:
            edge_index = _get_batch_edge_index(x, self.edge_index)
        elif self.batch_edge_index is None:
            self.batch_edge_index = _get_batch_edge_index(x, self.edge_index)
            edge_index = self.batch_edge_index
        else:
            edge_index = self.batch_edge_index

        # # x = self.prelu1(self.conv1(x, self.edge_index))
        # # x = self.prelu2(self.conv2(x, self.edge_index))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # x = self.conv2(x, edge_index)
        x = x.reshape(x.shape[0], 12 * 49)

        # x = x.flatten(1, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = th.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def _get_batch_edge_index(x, edge_index):
    return edge_index
    # return th.cat(tuple(edge_index for _ in range(x.shape[0])), dim=1)
    # return next(
    #     iter(
    #         DataLoader(
    #             [Data(x=x_, edge_index=edge_index) for x_ in x],
    #             batch_size=batch_size,
    #         )
    #     )
    # ).edge_index


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

    edge_index = HexBoard(
        "", size=policy_kwargs["board_size"], use_graph=True
    ).edge_index.to(device)
    mlp_size = policy_kwargs["net_arch"][0]
    actor = Actor(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    qf1 = QNetwork(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    qf2 = QNetwork(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    qf1_target = QNetwork(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    qf2_target = QNetwork(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    target_actor = Actor(env, mlp_size=mlp_size, edge_index=edge_index).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=LEARNING_RATE
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

    # fmt: off
    load_model_if_available(
        base_path,env_name,version,actor,target_actor,qf1,qf2,qf1_target,qf2_target,actor_optimizer,q_optimizer
    )  # fmt: on
    dtype = th.bfloat16  # TH_FLOAT_TYPE  # th.bfloat16
    optim_actor, optim_actor_optimizer = ipex.optimize(
        actor, optimizer=actor_optimizer, dtype=dtype
    )
    optim_actor = optim_actor.to(device)
    optim_qf1, optim_q_optimizer = ipex.optimize(
        qf1, optimizer=q_optimizer, dtype=dtype
    )
    q_optimizer_fake = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=LEARNING_RATE
    )
    optim_qf2, _ = ipex.optimize(qf2, optimizer=q_optimizer_fake, dtype=dtype)
    # optim_actor, optim_qf1, optim_qf2 = actor, qf1, qf2
    # optim_actor_optimizer, optim_q_optimizer = actor_optimizer, q_optimizer
    # compilation crashes, ipex triton requires torch 2.7 https://github.com/intel/intel-xpu-backend-for-triton
    # compiled_target_actor = th.compile(target_actor)
    # compiled_qf1 = th.compile(qf1)
    # compiled_qf1_target = th.compile(qf1_target)
    # compiled_qf2 = th.compile(qf2)
    # compiled_qf2_target = th.compile(qf2_target)
    try:
        # to train on bfloat16 optimize above for bfloat16
        context = th.amp.autocast(
            "xpu", enabled=dtype == th.bfloat16, dtype=th.bfloat16
        )
        # context = nullcontext()
        with context:
            train(
                env,
                optim_actor,
                optim_actor_optimizer,
                optim_q_optimizer,
                rb,
                writer,
                device,
                optim_qf1,
                optim_qf2,
                target_actor,
                qf1_target,
                qf2_target,
            )
    finally:
        save_models(
            new_path,
            env_name,
            version,
            actor,
            target_actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
        )

        env.close()
        writer.close()


# fmt: off
def load_model_if_available(
    base_path, env_name, version, actor, target_actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer
):  # fmt: on
    if os.path.exists(base_path):
        print("loading pre-trained weights...")
        actor.load_state_dict(
            th.load(f"{base_path}/{env_name}-td3-actor.v{version}", weights_only=True)
        )
        target_actor.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-td3-target-actor.v{version}", weights_only=True
            )
        )
        qf1.load_state_dict(
            th.load(f"{base_path}/{env_name}-td3-qf1.v{version}", weights_only=True)
        )
        qf2.load_state_dict(
            th.load(f"{base_path}/{env_name}-td3-qf2.v{version}", weights_only=True)
        )
        qf1_target.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-td3-target-qf1.v{version}", weights_only=True
            )
        )
        qf2_target.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-td3-target-qf2.v{version}", weights_only=True
            )
        )
        actor_optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-td3-actor-optimizer.v{version}",
                weights_only=True,
            )
        )
        q_optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-td3-q-optimizer.v{version}", weights_only=True
            )
        )

# fmt: off
def save_models(
    base_path, env_name, version, actor, target_actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer
):  # fmt: on
    th.save(actor.state_dict(), f"{base_path}/{env_name}-td3-actor.v{version + 1}")
    th.save(
        target_actor.state_dict(),
        f"{base_path}/{env_name}-td3-target-actor.v{version + 1}",
    )
    th.save(qf1.state_dict(), f"{base_path}/{env_name}-td3-qf1.v{version + 1}")
    th.save(qf2.state_dict(), f"{base_path}/{env_name}-td3-qf2.v{version + 1}")
    th.save(
        qf1_target.state_dict(),
        f"{base_path}/{env_name}-td3-target-qf1.v{version + 1}",
    )
    th.save(
        qf2_target.state_dict(),
        f"{base_path}/{env_name}-td3-target-qf2.v{version + 1}",
    )
    th.save(
        actor_optimizer.state_dict(),
        f"{base_path}/{env_name}-td3-actor-optimizer.v{version + 1}",
    )
    th.save(
        q_optimizer.state_dict(),
        f"{base_path}/{env_name}-td3-q-optimizer.v{version + 1}",
    )


def init_directory(env_name, version) -> tuple[str, str]:
    base_path = f"./{env_name}-td3.v{version}"
    new_path = f"./{env_name}-td3.v{version + 1}"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        try:
            with open(f"{new_path}/hyperparams.log", "w") as f:
                for param in [
                    TOTAL_TIMESTEPS,
                    LEARNING_RATE,
                    N_ENVS,
                    N_STEPS,
                    BUFFER_SIZE,
                    LEARNING_STARTS,
                    BATCH_SIZE,
                    GAMMA,
                    EXPLORATION_NOISE,
                    NOISE_CLIP,
                    POLICY_NOISE,
                    POLICY_FREQUENCY,
                    TAU,
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
    env: gym.vector.VectorEnv,
    actor: Actor,
    actor_optimizer,
    q_optimizer,
    rb: ReplayBuffer,
    writer,
    device,
    qf1,
    qf2,
    target_actor,
    qf1_target,
    qf2_target,
):
    obs, _ = env.reset(seed=1)
    for global_step in range(TOTAL_TIMESTEPS // N_ENVS):
        # ALGO LOGIC: put action logic here
        if global_step < LEARNING_STARTS // N_ENVS:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(N_ENVS)]
            )
        else:
            with th.no_grad():
                actions = actor(th.Tensor(obs).to(device))
                actions += th.normal(0, actor.action_scale * EXPLORATION_NOISE)
                actions = (
                    actions.cpu()
                    .numpy()
                    .clip(env.single_action_space.low, env.single_action_space.high)
                )

        # TRY NOT TO MODIFY: execute the game and log data.
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

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > LEARNING_STARTS // N_ENVS:
            data = rb.sample(BATCH_SIZE)
            with th.no_grad():
                clipped_noise = (
                    th.randn_like(data.actions, device=device) * POLICY_NOISE
                ).clamp(-NOISE_CLIP, NOISE_CLIP) * target_actor.action_scale

                next_state_actions = (
                    target_actor(data.next_observations) + clipped_noise
                ).clamp(env.single_action_space.low[0], env.single_action_space.high[0])
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = th.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * GAMMA * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % POLICY_FREQUENCY == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(
                    actor.parameters(), target_actor.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )
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

            if global_step % N_STEPS == 0:
                all_env_steps = global_step * N_ENVS
                writer.add_histogram(
                    "charts/actions",
                    np.clip(np.array(env.action_queue), -1, 1),
                    bins="auto",
                    max_bins=100,
                )
                writer.add_scalar(
                    "charts/episode_len", np.mean(env.length_queue), all_env_steps
                )
                writer.add_scalar(
                    "charts/episode_fps",
                    np.mean(env.time_queue) * N_ENVS,
                    all_env_steps,
                )
                writer.add_scalar(
                    "charts/mean_return", np.mean(env.return_queue), all_env_steps
                )
                writer.add_scalar(
                    "charts/mean_reward", np.mean(env.reward_queue), all_env_steps
                )
                writer.add_scalar(
                    "charts/mean_winner", np.mean(env.winner_queue), all_env_steps
                )
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), all_env_steps
                )
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), all_env_steps
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), all_env_steps
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), all_env_steps)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), all_env_steps)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, all_env_steps)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), all_env_steps)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                # writer.add_scalar(
                #     "charts/SPS",
                #     int(global_step / (time.time() - start_time)),
                #     global_step,
                # )
