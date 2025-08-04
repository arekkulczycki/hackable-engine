# -*- coding: utf-8 -*-
import os
import random
from collections import deque

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard
from hackable_engine.training.algorithms.util.replay_buffer import ReplayBuffer
from hackable_engine.training.utils.device import Device
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.hyperparams import *
from hackable_engine.training.models.mixins.actor_logit_mixin import ActorLogitMixin
from hackable_engine.training.models.graph_sg import GraphSG

LOG_PATH = "./hackable_engine/training/logs/"
TH_FLOAT_TYPE = th.float32

logit_queue = deque(maxlen=N_ENVS * 81)


class Critic(GraphSG): ...


class Actor(ActorLogitMixin, GraphSG): ...


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
    gnn_shape = policy_kwargs["gnn_arch"]
    mlp_shape = policy_kwargs["mlp_arch"]

    actor = Actor(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=BATCH_SIZE,
        num_envs=N_ENVS,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(device)
    qf1 = Critic(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=BATCH_SIZE,
        num_envs=N_ENVS,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(device)
    qf2 = Critic(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=BATCH_SIZE,
        num_envs=N_ENVS,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(device)
    qf1_target = Critic(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=BATCH_SIZE,
        num_envs=N_ENVS,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(device)
    qf2_target = Critic(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=BATCH_SIZE,
        num_envs=N_ENVS,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=Q_LEARNING_RATE
    )
    q_optimizer_scheduler = LambdaLR(q_optimizer, get_learning_rate_decay())
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=LEARNING_RATE)
    actor_optimizer_scheduler = LambdaLR(actor_optimizer, get_learning_rate_decay())
    print("allocated MB", th.xpu.memory_allocated() / 1024 / 1024)

    env.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        BUFFER_SIZE,
        env.single_observation_space,
        env.single_action_space,
        # gym.spaces.Box(low=0, high=80, shape=(1,)),
        # gym.spaces.Box(low=0, high=1, shape=(81,)),
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

    def one(step):
        return 1

    return one


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
    for global_step in range(TOTAL_TIMESTEPS):
        if global_step < LEARNING_STARTS:
            actions = np.array(
                [env.single_action_space.sample() for _ in range(N_ENVS)]
            )
        else:
            th_obs = th.tensor(obs).to(device)
            actions, _, _, logits = actor.get_action(th_obs)
            logit_queue.extend(logits.detach().cpu().flatten().numpy())
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

        if global_step > LEARNING_STARTS:
            data = rb.sample(BATCH_SIZE)
            with th.no_grad():
                _, next_state_log_pi, next_state_action_probs, _ = actor.get_action(
                    data.next_observations
                )
                qf1_next_target = qf1_target(data.next_observations)
                qf2_next_target = qf2_target(data.next_observations)

                min_qf_next_target = next_state_action_probs * (
                    th.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                )

                gamma = GAMMA
                if not np.any(terminations):
                    mean_reward = (rewards * terminations)[terminations == 1].mean().item()
                    a = min(max(mean_reward + 1.5, 0), 1)
                    gamma = GAMMA * a + 0.05 * (1 - a)

                min_qf_next_target = min_qf_next_target.sum(dim=1)
                next_q_value = data.rewards.flatten() + (
                    1 - data.dones.flatten()
                ) * gamma * (min_qf_next_target)

            qf1_values = qf1(data.observations)
            qf2_values = qf2(data.observations)
            qf1_values_gathered = qf1_values.gather(1, data.actions.long())
            qf2_values_gathered = qf2_values.gather(1, data.actions.long())
            qf1_a_values = qf1_values_gathered.view(-1)
            qf2_a_values = qf2_values_gathered.view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            q_optimizer.zero_grad()
            qf_loss.backward()

            q_optimizer.step()
            q_optimizer_scheduler.step()

            if global_step % POLICY_FREQUENCY == 0:
                for _ in range(POLICY_FREQUENCY):
                    _, log_pi, action_probs, _ = actor.get_action(data.observations)

                    with th.no_grad():
                        qf1_values = qf1(data.observations)
                        qf2_values = qf2(data.observations)
                        min_qf_values = th.min(qf1_values, qf2_values)

                    actor_loss = (
                        action_probs * ((alpha * log_pi) - min_qf_values)
                    ).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    actor_optimizer_scheduler.step()

                    if ENTROPY_AUTOTUNE:
                        alpha_loss = (
                            action_probs.detach()
                            * (-log_alpha.exp() * (log_pi + target_entropy).detach())
                        ).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

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

            if global_step % 100 == 0:
                all_env_steps = global_step * N_ENVS
                env_progress_data: EnvProgressData = env.get_progress_data()
                logit_array = np.array(logit_queue)
                if logit_array.size > 1000:
                    writer.add_histogram(
                        "charts/actions",
                        logit_array,
                        bins="auto",
                        max_bins=100,
                    )
                writer.add_scalar(
                    "charts/episode_len", env_progress_data.length_mean, all_env_steps
                )
                writer.add_scalar(
                    "charts/episode_fps",
                    env_progress_data.time_mean * N_ENVS,
                    all_env_steps,
                )
                writer.add_scalar(
                    "charts/mean_return", env_progress_data.return_mean, all_env_steps
                )
                writer.add_scalar(
                    "charts/mean_winner", env_progress_data.winner_mean, all_env_steps
                )
                writer.add_scalar(
                    "charts/mean_reward", env_progress_data.reward_mean, all_env_steps
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
                # writer.add_scalar("misc/entropy(std)", std.mean(), all_env_steps)
                writer.add_scalar(
                    "misc/actor_gradient",
                    calculate_gradient(actor.parameters()),
                    all_env_steps,
                )
                writer.add_scalar(
                    "misc/q1_gradient",
                    calculate_gradient(qf1.parameters()),
                    all_env_steps,
                )
                writer.add_scalar(
                    "misc/q2_gradient",
                    calculate_gradient(qf2.parameters()),
                    all_env_steps,
                )
                writer.add_scalar(
                    "misc/lr",
                    actor_optimizer_scheduler.get_last_lr()[0],
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
            gradients.append(param.grad.view(-1))  # Flatten the gradients
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
