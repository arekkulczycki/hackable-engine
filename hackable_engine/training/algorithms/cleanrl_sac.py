# -*- coding: utf-8 -*-
import os
import random
from time import perf_counter

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from gymnasium.vector import VectorEnv
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import SGConv, LayerNorm, GraphNorm, GATConv

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import TH_FLOAT_TYPE
from hackable_engine.training.device import Device
from hackable_engine.training.hyperparams import *

LOG_PATH = "./hackable_engine/training/logs/"
LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(
        self,
        env,
        board_size,
        mlp_size,
        edge_index,
        edge_types,
        should_extract_features=False,
    ):
        super().__init__()
        self.env = env
        self.board_size = board_size

        self.should_extract_features = should_extract_features
        if should_extract_features:
            shape = [18, 36, 72]
            # l1 = nn.Linear(1, shape[0])
            # l2 = nn.Linear(shape[0], shape[0])
            # for layer in (l1, l2):
            #     th.nn.init.kaiming_normal_(
            #         layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            #     )
            #     th.nn.init.zeros_(layer.bias)
            #
            # self.feature_expansion = nn.Sequential(l1, nn.LeakyReLU(), l2)
            self.feature_expansion = nn.Embedding(
                3, shape[0], dtype=TH_FLOAT_TYPE, device=Device.XPU
            )
            self.setup_graph_feature_extractor(edge_index, edge_types, shape)

        self.fc1 = nn.Linear(
            (
                shape[-1] + np.prod(env.single_action_space.shape)
                if self.should_extract_features
                else (
                    np.array(env.single_observation_space.shape).prod()
                    + np.prod(env.single_action_space.shape)
                )
            ),
            mlp_size[0],
            device=Device.XPU,
        )
        self.fc2 = nn.Linear(mlp_size[0], mlp_size[1], device=Device.XPU)
        self.fc3 = nn.Linear(mlp_size[1], 1, device=Device.XPU)

        self.initialize_fc_weights()

    def setup_graph_feature_extractor(self, edge_index, edge_types, shape):
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.batch_small = th.repeat_interleave(
            th.arange(N_ENVS), self.board_size**2
        ).to(Device.XPU)
        self.batch_small_exp = self.batch_small.unsqueeze(-1).expand(-1, shape[-1])
        self.batch_big = th.repeat_interleave(
            th.arange(BATCH_SIZE), self.board_size**2
        ).to(Device.XPU)
        self.batch_big_exp = self.batch_big.unsqueeze(-1).expand(-1, shape[-1])
        self.scatter_dummy_small = th.zeros((N_ENVS, shape[-1])).to(Device.XPU)
        self.scatter_dummy_big = th.zeros((BATCH_SIZE, shape[-1])).to(Device.XPU)

        # self.res_proj_0 = nn.Linear(1, shape[0])
        self.res_proj_1 = nn.Linear(shape[0], shape[1], device=Device.XPU)
        self.res_proj_2 = nn.Linear(shape[1], shape[2], device=Device.XPU)
        self.conv1 = SGConv(shape[0], shape[0], K=1)
        self.conv2 = SGConv(shape[0], shape[1], K=1)
        self.conv3 = SGConv(shape[1], shape[2], K=1)
        # self.conv1 = GATConv(shape[0], shape[0], heads=4, concat=False)
        # self.conv2 = GATConv(shape[0], shape[1], heads=4, concat=False)
        # self.conv3 = GATConv(shape[1], shape[2], heads=4, concat=False)
        self.norm1 = GraphNorm(shape[0])
        self.norm2 = GraphNorm(shape[1])
        self.norm3 = GraphNorm(shape[2])

    def forward(self, x, a):
        if self.should_extract_features:
            # batch_size, num_nodes, _ = x.shape
            # x = x.view(-1, 1)

            x = x.flatten()
            x = self.feature_expansion((x + 1).int())
            # x = x.view(batch_size, num_nodes, -1)
            x = self.extract_features(x)
        else:
            x = x.flatten(1, -1)

        x = th.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def extract_features(self, x):
        batch, batch_exp, batch_size, scatter_dummy = (
            (
                self.batch_small,
                self.batch_small_exp,
                N_ENVS,
                self.scatter_dummy_small,
            )
            if x.shape[0] == N_ENVS * self.board_size**2
            else (
                self.batch_big,
                self.batch_big_exp,
                BATCH_SIZE,
                self.scatter_dummy_big,
            )
        )

        # x = x.flatten(0, 1)
        res0 = x  # self.res_proj_0(x)
        x = self.conv1(x, self.edge_index)
        x = self.norm1(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res0
        res1 = self.res_proj_1(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = self.conv2(x, self.edge_index)
        x = self.norm2(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res1
        res2 = self.res_proj_2(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = self.conv3(x, self.edge_index)
        x = self.norm3(x, batch, batch_size)
        x = x + res2
        x = th.scatter_reduce(
            scatter_dummy, 0, batch_exp, reduce="mean", src=x, include_self=False
        )  # TODO: use segment_reduce when supported
        # t2 = perf_counter()
        return x

    def initialize_fc_weights(self):
        for layer in (self.fc1, self.fc2, self.fc3):
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            th.nn.init.zeros_(layer.bias)


class Actor(nn.Module):
    def __init__(
        self,
        env: VectorEnv,
        board_size,
        mlp_size,
        edge_index,
        edge_types,
        should_extract_features=False,
        add_auxiliary_prediction=False,
    ):
        super().__init__()
        self.env = env
        self.board_size = board_size

        self.should_extract_features = should_extract_features
        self.add_auxiliary_prediction = add_auxiliary_prediction
        if should_extract_features:
            shape = [18, 36, 72]
            # l1 = nn.Linear(1, shape[0])
            # l2 = nn.Linear(shape[0], shape[0])
            # for layer in (l1, l2):
            #     th.nn.init.kaiming_normal_(
            #         layer.weight, mode="fan_in", nonlinearity="leaky_relu"
            #     )
            #     th.nn.init.zeros_(layer.bias)
            #
            # self.feature_expansion = nn.Sequential(l1, nn.LeakyReLU(), l2)
            self.feature_expansion = nn.Embedding(
                3, shape[0], dtype=TH_FLOAT_TYPE, device=Device.XPU
            )
            self.setup_graph_feature_extractor(edge_index, edge_types, shape)

            if add_auxiliary_prediction:
                self.setup_auxiliary_fc(env, shape[-1])

        self.fc1 = nn.Linear(
            (
                shape[-1]
                if self.should_extract_features
                else np.array(env.single_observation_space.shape).prod()
            ),
            mlp_size[0],
            device=Device.XPU,
        )
        self.fc2 = nn.Linear(mlp_size[0], mlp_size[1], device=Device.XPU)
        self.fc_mean = nn.Linear(mlp_size[1], 1, device=Device.XPU)
        self.fc_logstd = nn.Linear(mlp_size[1], 1, device=Device.XPU)
        # action rescaling
        self.register_buffer(
            "action_scale",
            th.tensor(
                # (env.single_action_space.high - env.single_action_space.low) / 2.0,
                1.0,
                dtype=th.float32,
                device=Device.XPU,
            ),
        )
        self.register_buffer(
            "action_bias",
            th.tensor(
                # (env.single_action_space.high + env.single_action_space.low) / 2.0,
                0.0,
                dtype=th.float32,
                device=Device.XPU,
            ),
        )

        self.initialize_fc_weights()

    def setup_graph_feature_extractor(self, edge_index, edge_types, shape):
        self.edge_index = edge_index
        self.edge_types = edge_types
        self.batch_small = th.repeat_interleave(
            th.arange(N_ENVS), self.board_size**2
        ).to(Device.XPU)
        self.batch_small_exp = self.batch_small.unsqueeze(-1).expand(-1, shape[-1])
        self.batch_big = th.repeat_interleave(
            th.arange(BATCH_SIZE), self.board_size**2
        ).to(Device.XPU)
        self.batch_big_exp = self.batch_big.unsqueeze(-1).expand(-1, shape[-1])
        self.scatter_dummy_small = th.zeros((N_ENVS, shape[-1])).to(Device.XPU)
        self.scatter_dummy_big = th.zeros((BATCH_SIZE, shape[-1])).to(Device.XPU)

        # self.res_proj_0 = nn.Linear(1, shape[0])
        self.res_proj_1 = nn.Linear(shape[0], shape[1], device=Device.XPU)
        self.res_proj_2 = nn.Linear(shape[1], shape[2], device=Device.XPU)
        self.conv1 = SGConv(shape[0], shape[0], K=1)
        self.conv2 = SGConv(shape[0], shape[1], K=1)
        self.conv3 = SGConv(shape[1], shape[2], K=1)
        # self.conv1 = GATConv(shape[0], shape[0], heads=4, concat=False)
        # self.conv2 = GATConv(shape[0], shape[1], heads=4, concat=False)
        # self.conv3 = GATConv(shape[1], shape[2], heads=4, concat=False)
        self.norm1 = GraphNorm(shape[0])
        self.norm2 = GraphNorm(shape[1])
        self.norm3 = GraphNorm(shape[2])

    def setup_auxiliary_fc(self, env, conv_out):
        self.afc1 = nn.Linear(
            (
                conv_out
                if self.should_extract_features
                else np.array(env.single_observation_space.shape).prod()
            ),
            16,
        )
        self.afc2 = nn.Linear(16, 16)
        self.afc3 = nn.Linear(16, 2)
        self.afc = nn.Sequential(self.afc1, self.afc2, self.afc3, nn.Tanh())

    def forward(self, x):
        # t0 = perf_counter()
        if self.should_extract_features:
            # batch_size, num_nodes, _ = x.shape
            # x = x.view(-1, 1)
            x = x.flatten()
            x = self.feature_expansion((x + 1).int())
            # x = x.view(batch_size, num_nodes, -1)
            x = self.extract_features(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        aux_actions = None
        if self.add_auxiliary_prediction:
            aux_actions = self.afc(x)

        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = th.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        # t3 = perf_counter()
        # print("forward took", t3 - t0)#, t2 - t1, t1 - t0)
        return mean, log_std, aux_actions

    def extract_features(self, x):
        batch, batch_exp, batch_size, scatter_dummy = (
            (
                self.batch_small,
                self.batch_small_exp,
                N_ENVS,
                self.scatter_dummy_small,
            )
            if x.shape[0] == N_ENVS * self.board_size**2
            else (
                self.batch_big,
                self.batch_big_exp,
                BATCH_SIZE,
                self.scatter_dummy_big,
            )
        )

        # x = x.flatten(0, 1)
        res0 = x  # self.res_proj_0(x)
        x = self.conv1(x, self.edge_index)
        x = self.norm1(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res0
        res1 = self.res_proj_1(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = self.conv2(x, self.edge_index)
        x = self.norm2(x, batch, batch_size)
        # x = F.dropout(x, p=0.2)
        x = x + res1
        res2 = self.res_proj_2(x)
        # x = F.relu(F.dropout(x, p=0.2))
        x = self.conv3(x, self.edge_index)
        x = self.norm3(x, batch, batch_size)
        x = x + res2
        x = th.scatter_reduce(
            scatter_dummy, 0, batch_exp, reduce="amax", src=x, include_self=False
        )  # TODO: use segment_reduce when supported
        # t2 = perf_counter()
        return x

    def get_action(self, x):
        if not self.should_extract_features:
            x = x.flatten(1, -1)
        # t0 = perf_counter()
        mean, log_std, aux_actions = self(x)
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
        # t3 = perf_counter()
        # print("get action", t3-t0)
        # TODO: this is for LogNormal, erase for Normal distribution
        # action = action * 2 - 1
        if self.add_auxiliary_prediction:
            return th.cat((action, aux_actions), dim=1), log_prob, mean
        return action, log_prob, mean, std

    def initialize_fc_weights(self):
        for layer in (self.fc1, self.fc2, self.fc_mean, self.fc_logstd):
            th.nn.init.kaiming_normal_(
                layer.weight, mode="fan_out", nonlinearity="relu"
            )
            th.nn.init.zeros_(layer.bias)


def run(version, policy_kwargs, env, env_name, device):
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

    board = HexBoard("", size=policy_kwargs["board_size"], use_graph=True)
    edge_index = board.edge_index.to(device)
    edge_types = board.edge_types.to(device)
    mlp_size = policy_kwargs["net_arch"]

    actor = Actor(
        env,
        board.size,
        mlp_size=mlp_size,
        edge_index=edge_index.clone(),
        edge_types=edge_types,
    ).to(device)
    qf1 = SoftQNetwork(
        env,
        board.size,
        mlp_size=mlp_size,
        edge_index=edge_index.clone(),
        edge_types=edge_types,
    ).to(device)
    qf2 = SoftQNetwork(
        env,
        board.size,
        mlp_size=mlp_size,
        edge_index=edge_index.clone(),
        edge_types=edge_types,
    ).to(device)
    qf1_target = SoftQNetwork(
        env,
        board.size,
        mlp_size=mlp_size,
        edge_index=edge_index.clone(),
        edge_types=edge_types,
    ).to(device)
    qf2_target = SoftQNetwork(
        env,
        board.size,
        mlp_size=mlp_size,
        edge_index=edge_index.clone(),
        edge_types=edge_types,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=Q_LEARNING_RATE
    )
    q_optimizer_scheduler = LambdaLR(
        q_optimizer, get_learning_rate_decay()
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=LEARNING_RATE)
    actor_optimizer_scheduler = LambdaLR(
        actor_optimizer, get_learning_rate_decay()
    )
    print("allocated", th.xpu.memory_allocated() / 1024 / 1024)

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
        base_path,env_name,version,actor,qf1,qf2,qf1_target,qf2_target,actor_optimizer,q_optimizer
    )  # fmt: on
    print("starting")

    if TH_FLOAT_TYPE is th.bfloat16:
        compile_models(
            actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
            th.float32,
        )
    # for element in [actor, qf1, qf2, qf1_target, qf2_target]:#, actor_optimizer, q_optimizer]:
    #     element.to(Device.XPU)
    # print(th.xpu.memory_allocated() / 1024 / 1024)
    # actor = th.compile(actor, dynamic=False)
    # qf1 = th.compile(qf1, dynamic=False)
    # qf1_target = th.compile(qf1_target, dynamic=False)
    # qf2 = th.compile(qf2, dynamic=False)
    # qf2_target = th.compile(qf2_target, dynamic=False)
    try:
        # context = (
        #     th.amp.autocast(Device.XPU.value, enabled=True, dtype=th.bfloat16)
        #     if device == Device.XPU
        #     else nullcontext()
        # )
        with th.amp.autocast(
            Device.XPU.value, enabled=TH_FLOAT_TYPE is th.bfloat16, dtype=th.float16
        ):
            # with nullcontext():
            train(
                env,
                actor,
                actor_optimizer,
                actor_optimizer_scheduler,
                q_optimizer,
                q_optimizer_scheduler,
                rb,
                writer,
                device,
                qf1,
                qf2,
                qf1_target,
                qf2_target,
            )
    finally:
        save_models(
            new_path,
            env_name,
            version,
            actor,
            qf1,
            qf2,
            qf1_target,
            qf2_target,
            actor_optimizer,
            q_optimizer,
        )

        env.close()
        writer.close()


def load_model_if_available(
    base_path,
    env_name,
    version,
    actor,
    qf1,
    qf2,
    qf1_target,
    qf2_target,
    actor_optimizer,
    q_optimizer,
):  # fmt: on
    if os.path.exists(base_path):
        print("loading pre-trained weights...")
        actor.load_state_dict(
            th.load(f"{base_path}/{env_name}-sac-actor.v{version}", weights_only=True)
        )
        qf1.load_state_dict(
            th.load(f"{base_path}/{env_name}-sac-qf1.v{version}", weights_only=True)
        )
        qf2.load_state_dict(
            th.load(f"{base_path}/{env_name}-sac-qf2.v{version}", weights_only=True)
        )
        qf1_target.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-sac-target-qf1.v{version}", weights_only=True
            )
        )
        qf2_target.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-sac-target-qf2.v{version}", weights_only=True
            )
        )
        actor_optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-sac-actor-optimizer.v{version}",
                weights_only=True,
            )
        )
        q_optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-sac-q-optimizer.v{version}", weights_only=True
            )
        )


# fmt: off
def save_models(
    new_path, env_name, version, actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer
):  # fmt: on
    th.save(actor.state_dict(), f"{new_path}/{env_name}-sac-actor.v{version + 1}")
    th.save(qf1.state_dict(), f"{new_path}/{env_name}-sac-qf1.v{version + 1}")
    th.save(qf2.state_dict(), f"{new_path}/{env_name}-sac-qf2.v{version + 1}")
    th.save(
        qf1_target.state_dict(),
        f"{new_path}/{env_name}-sac-target-qf1.v{version + 1}",
    )
    th.save(
        qf2_target.state_dict(),
        f"{new_path}/{env_name}-sac-target-qf2.v{version + 1}",
    )
    th.save(
        actor_optimizer.state_dict(),
        f"{new_path}/{env_name}-sac-actor-optimizer.v{version + 1}",
    )
    th.save(
        q_optimizer.state_dict(),
        f"{new_path}/{env_name}-sac-q-optimizer.v{version + 1}",
    )


def init_directory(env_name, version) -> tuple[str, str]:
    base_path = f"./{env_name}-sac.v{version}"
    new_path = f"./{env_name}-sac.v{version + 1}"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        try:
            with open(f"{new_path}/hyperparams.log", "w") as f:
                for param in [
                    TOTAL_TIMESTEPS,
                    LEARNING_RATE,
                    Q_LEARNING_RATE,
                    N_ENVS,
                    BUFFER_SIZE,
                    LEARNING_STARTS,
                    BATCH_SIZE,
                    GAMMA,
                    POLICY_FREQUENCY,
                    TARGET_NETWORK_FREQUENCY,
                    TAU,
                    ENTROPY_AUTOTUNE,
                    ENTROPY_ALPHA,
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


def get_learning_rate_decay():
    def wrapped(step):
        warm_up = 0.05
        if step / BUFFER_SIZE < warm_up:
            return 1 - (BUFFER_SIZE * warm_up - step) / (BUFFER_SIZE * warm_up)
        return max((1 - step / ((1 - warm_up) * BUFFER_SIZE)), 0.01)

    return wrapped


def train(
    env: gym.vector.VectorEnv,
    actor: Actor,
    actor_optimizer,
    actor_optimizer_scheduler,
    q_optimizer,
    q_optimizer_scheduler,
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
        alpha = max(log_alpha.exp().item(), MIN_ENTROPY_ALPHA)
        a_optimizer = optim.Adam([log_alpha], lr=ALPHA_LEARNING_RATE)
    else:
        alpha = ENTROPY_ALPHA

    obs, _ = env.reset(seed=1)
    for global_step in range(TOTAL_TIMESTEPS // N_ENVS):
        # ALGO LOGIC: put action logic here
        if global_step < LEARNING_STARTS // N_ENVS:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(N_ENVS)]
            )
        else:
            actions, _, _, _ = actor.get_action(th.tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        # TODO: implement Prioritized Replay, i.e.
        #  add to the buffer with weights proportional to target q errors
        #  `priorities = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())`
        obs = next_obs  # for the next iteration

        if global_step > LEARNING_STARTS // N_ENVS:
            # t0 = perf_counter()
            data = rb.sample(BATCH_SIZE)
            with th.no_grad():
                next_state_actions, next_state_log_pi, _, _ = actor.get_action(
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

            # t1 = perf_counter()
            # t = t1 - t0
            # if t > 0.5:
            #     print("1 took", t)
            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()
            # TODO: check if still clipping still needed for gradient explosion
            # th.nn.utils.clip_grad_norm_(qf1.parameters(), MAX_GRAD_NORM)
            # th.nn.utils.clip_grad_norm_(qf2.parameters(), MAX_GRAD_NORM)
            q_optimizer.step()
            q_optimizer_scheduler.step()

            # t2 = perf_counter()
            # t = t2 - t1
            # if t > 0.5:
            #     print("2 took", t)
            if global_step % POLICY_FREQUENCY == 0:  # TD 3 Delayed update support
                for _ in range(
                    POLICY_FREQUENCY
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _, std = actor.get_action(data.observations)

                    # t30 = perf_counter()

                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = th.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    # t31 = perf_counter()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    # TODO: check if still clipping still needed for gradient explosion
                    # th.nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
                    actor_optimizer.step()
                    actor_optimizer_scheduler.step()

                    # t32 = perf_counter()

                    if ENTROPY_AUTOTUNE:
                        with th.no_grad():
                            _, log_pi, _, _ = actor.get_action(data.observations)
                        alpha_loss = (
                            -log_alpha.exp() * (log_pi + target_entropy)
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = max(log_alpha.exp().item(), MIN_ENTROPY_ALPHA)

                    # t33 = perf_counter()
                    # print("took", t33 - t32, t32 - t31, t31 - t30, t30 - t2)
            # t3 = perf_counter()
            # t = t3 - t2
            # if t > 0.5:
            #     print("3 took", t, t3full-t2, t30-t3full, t31-t30, t32-t31)
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
            # t4 = perf_counter()
            # t = t4 - t3
            # if t > 0.5:
            #     print("4 took", t)
            if global_step % 100 == 0:
                all_env_steps = global_step * N_ENVS
                action_array = np.array(env.action_queue)
                try:
                    writer.add_histogram(
                        "charts/actions",
                        np.array(env.action_queue),
                        bins="auto",
                        max_bins=100,
                    )
                except:
                    print("histogram failed, action array size:", action_array.shape)
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
                    "charts/mean_winner", np.mean(env.winner_queue), all_env_steps
                )
                writer.add_scalar(
                    "charts/mean_reward", np.mean(env.reward_queue), all_env_steps
                )
                # writer.add_scalar("charts/mean_reward", np.mean(final_rewards), global_step)
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
                writer.add_scalar("losses/alpha", alpha, all_env_steps)
                writer.add_scalar("misc/entropy(std)", std.mean(), all_env_steps)
                writer.add_scalar("misc/actor_gradient", calculate_gradient(actor.parameters()), all_env_steps)
                writer.add_scalar("misc/q1_gradient", calculate_gradient(qf1.parameters()), all_env_steps)
                writer.add_scalar("misc/q2_gradient", calculate_gradient(qf2.parameters()), all_env_steps)
                writer.add_scalar(
                    "misc/lr",
                    get_learning_rate_decay()(global_step),
                    all_env_steps,
                )
                # print("SPS:", int(global_step / (time.time() - start_time)))
                # writer.add_scalar(
                #     "charts/SPS",
                #     int(global_step / (time.time() - start_time)),
                #     global_step,
                # )
                if ENTROPY_AUTOTUNE:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), all_env_steps
                    )


def calculate_gradient(params):
    gradients = []
    for param in params:
        if param.grad is not None:
            gradients.append(
                param.grad.view(-1)
            )  # Flatten the gradients
    gradients = th.cat(gradients)  # Concatenate all gradients
    return gradients.norm()


def compile_models(
    actor, qf1, qf2, qf1_target, qf2_target, actor_optimizer, q_optimizer, dtype
):
    qf2_optimizer_fake = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=LEARNING_RATE
    )
    qf1t_optimizer_fake = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=LEARNING_RATE
    )
    qf2t_optimizer_fake = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=LEARNING_RATE
    )

    th.xpu.optimize(
        actor,
        optimizer=actor_optimizer,
        inplace=True,
        graph_mode=True,
        dtype=dtype,
    )
    # optim_actor = optim_actor.to(device)
    th.xpu.optimize(
        qf1, optimizer=q_optimizer, inplace=True, graph_mode=True, dtype=dtype
    )
    th.xpu.optimize(
        qf2,
        optimizer=qf2_optimizer_fake,
        inplace=True,
        graph_mode=True,
        dtype=dtype,
    )
    th.xpu.optimize(
        qf1_target,
        optimizer=qf1t_optimizer_fake,
        inplace=True,
        graph_mode=True,
        dtype=dtype,
    )
    th.xpu.optimize(
        qf2_target,
        optimizer=qf2t_optimizer_fake,
        inplace=True,
        graph_mode=True,
        dtype=dtype,
    )
