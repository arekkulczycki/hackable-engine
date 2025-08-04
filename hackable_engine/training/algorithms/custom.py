# -*- coding: utf-8 -*-
import os

from torch.optim.lr_scheduler import LambdaLR
from collections import deque
from itertools import batched
from multiprocessing import Queue, Process
from random import sample
from time import perf_counter
from typing import Iterable

import gymnasium as gym
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.norm import GraphNorm

from hackable_engine.board.hex.bitboard_utils import generate_masks
from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard
from hackable_engine.board.hex.move import Move
from hackable_engine.training.utils.device import Device
from hackable_engine.training.envs.hex.logit_9_graph_env import Logit9GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import MultiprocessAsyncEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_env import MultiprocessEnv
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_sg import GraphSG
from hackable_engine.training.run import LOG_PATH

n_workers = 4
n_envs = 64
replay_batch_size = 128
replay_runs = 9
"""This many times trains on replay for each 1 new experience."""
replay_buffer_size = replay_batch_size * replay_runs * 1000
total_steps = 10_000_000
lr = 6e-4
gamma = 0.9
# cross_ent_coef = 0.1
# advantage_coef = 100
value_coef = 0.75
ent_coef = 0.0005
"""Entropy bonus to avoid action saturation around 0 and 1"""
ent_coef_shift = 0.75
"""0 to 1. Shifts when negative entropy loss flips sign to positive. 
0 means mid-training, 1 will start with positive"""
action_queue = deque(maxlen=25 * n_envs)
max_action_queue = deque(maxlen=n_envs)
policy_loss_queue = deque(maxlen=5 * n_envs)
policy_value_loss_queue = deque(maxlen=5 * n_envs)
value_loss_queue = deque(maxlen=5 * n_envs)
entropy_queue = deque(maxlen=5 * n_envs)


class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, batch):
        # unpacks the batches to shuffle the replays
        self.buffer.extend(zip(*batch))

    def sample(self, batch_size: int):
        batch = sample(self.buffer, batch_size)
        obss, n_movess, best_moves, best_scores, rewards = zip(*batch)
        return (
            th.stack(tuple(th.from_numpy(obs) for obs in obss)).to(Device.XPU),
            n_movess,
            best_moves,
            best_scores,
            rewards,
        )

    def size(self):
        return len(self.buffer)


def prepare_model():
    hetero_graph_embedding = board.get_hetero_graph_node_embedding()
    _, white_edge, black_edge = th.split(
        th.from_numpy(hetero_graph_embedding).to(th.int), 1
    )
    white_edge = white_edge.flatten(0, 1)
    black_edge = black_edge.flatten(0, 1)
    white_edge_replay = white_edge.clone()
    black_edge_replay = black_edge.clone()
    if n_envs > 1:
        white_edge = th.stack([white_edge for _ in range(n_envs)], dim=0)
        black_edge = th.stack([black_edge for _ in range(n_envs)], dim=0)
        white_edge_replay = th.stack(
            [white_edge_replay for _ in range(replay_batch_size)], dim=0
        )
        black_edge_replay = th.stack(
            [black_edge_replay for _ in range(replay_batch_size)], dim=0
        )

    edge_types = board.edge_types.to(th.int)
    edge_index = board.edge_index.to(th.int64)
    batch_index = th.repeat_interleave(th.arange(n_envs), board.size_square)
    batch_index_replay = th.repeat_interleave(
        th.arange(replay_batch_size), board.size_square
    )

    gnn_shape = (36, 108, 216, 216)
    mlp_shape = (128,)
    batch_size = 64
    actor_model = GraphSG(
        node_count=board.size_square,
        node_features=3,
        output_size=board.size_square,
        batch_size=batch_size,
        num_envs=16,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(Device.XPU)
    critic_model = GraphSG(
        node_count=board.size_square,
        node_features=3,
        output_size=1,
        batch_size=batch_size,
        num_envs=16,
        gnn_shape=gnn_shape,
        mlp_shape=mlp_shape,
        edge_index=board.edge_index,
    ).to(Device.XPU)
    actor_model.train()
    critic_model.train()
    actor_optimizer = th.optim.Adam(actor_model.parameters(), lr=lr)
    critic_optimizer = th.optim.Adam(critic_model.parameters(), lr=lr * 1.5)
    # optimizer = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # optimizer = th.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return actor_model, critic_model, actor_optimizer, critic_optimizer


def get_learning_rate_decay():
    def wrapped(step):
        warm_up = 0.05
        if step / total_steps < warm_up:
            return 1 - (total_steps * warm_up - step) / (total_steps * warm_up)
        return  max((1 - step / ((1 - warm_up) * total_steps)) ** 2, 0.01)

    return wrapped


def get_adjusted_entropy(coef, step):
    """Start negative, later turn positive"""
    return (2 * step / total_steps - 1 + ent_coef_shift) * coef


def run(writer):
    env = MultiprocessEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            gym.make_vec(
                id=Logit9GraphEnv.__name__,
                num_envs=num_envs,
                color=color_,
                models=models,
            ),
            is_multiprocessed=True,
        ),
        2,
        8,
        action_shape=(1,),
        color=False,
    )
    obs, _ = env.reset()

    replay_buffer = ReplayBuffer(replay_buffer_size)
    for iteration, step in enumerate(
        range(1, total_steps + 1, n_envs * (1 + replay_runs))
    ):
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        # t0 = perf_counter()

        node_colors = th.from_numpy(obs).to(Device.XPU)
        batched_probs, batched_probs_clone, max_probs, value = run_model(node_colors)

        # t1 = perf_counter()
        # print("1", t1 - t0)

        best_moves, best_scores, n_movess, rewards = run_env(
            env, replay_buffer, batched_probs_clone
        )

        # t2 = perf_counter()
        # print("2", t2 - t1)

        train(
            n_movess,
            best_moves,
            best_scores,
            rewards,
            batched_probs,
            value,
            max_probs,
            step,
            n_envs,
        )

        # t3 = perf_counter()
        # print("3", t3 - t2)

        if iteration > 10 * replay_runs:
            train_on_replay(replay_buffer, step)

        if iteration % 10 == 0:
            actor_gradients = []
            for param in actor_model.parameters():
                if param.grad is not None:
                    actor_gradients.append(param.grad.view(-1))
            actor_gradients = th.cat(actor_gradients)
            actor_gradient = actor_gradients.norm()

            critic_gradients = []
            for param in critic_model.parameters():
                if param.grad is not None:
                    critic_gradients.append(param.grad.view(-1))
            critic_gradients = th.cat(critic_gradients)
            critic_gradient = critic_gradients.norm()

            log_progress(env, step, actor_gradient, critic_gradient)


def run_model(x):
    policy_logits = actor_model(x)
    value = critic_model(x)
    batch_size = value.shape[0]
    batched_logits = policy_logits.reshape((batch_size, board.size_square))
    batched_probs = th.nn.functional.softmax(batched_logits, dim=1)
    max_probs = th.tensor([th.max(probs) for probs in batched_probs])

    batched_probs_clone = batched_probs.detach().cpu()

    max_action_queue.extend(max_probs)
    action_queue.extend(batched_probs_clone.flatten().tolist())

    return batched_probs, batched_probs_clone, max_probs, value


def run_env(env, replay_buffer: ReplayBuffer, batched_probs_clone):
    env_data = [(None, 0.0, 0) for _ in range(n_envs)]
    jobs = (
        (i, env_.controller.board)
        for i, env_ in enumerate((e.unwrapped for e in env.unwrapped.envs))
        if env_.winner is None
    )

    ii, boards = zip(*jobs)
    results = collect_worker_responses(batched(boards, n_workers))
    for i, result in zip(ii, results):
        env_data[i] = result

    best_moves, best_scores, n_movess = list(zip(*env_data))

    # t3 = perf_counter()
    # print("3", t3 - t2)
    # TODO: instead of deterministic choice we can choose action from a distribution
    #  although in such case the action might be an illegal move
    # generate_unoccupied_from_distrib(board.size, batched_probs_clone, boards)
    move_positions = get_moves_from_probs(batched_probs_clone)
    obs, rewards, dones, _, infos = env.step(move_positions)
    replay_buffer.add((obs, n_movess, best_moves, best_scores, rewards))

    return best_moves, best_scores, n_movess, rewards


def train(
    n_movess,
    best_moves,
    best_scores,
    rewards,
    batched_probs,
    value,
    max_probs,
    step,
    batch_size,
):
    # target policy weights sum to 1, as if it was a result of softmax
    target_policy = th.zeros((batch_size, board.size_square)) + 0.1 / (
        board.size_square - 1
    )
    for i, best_move in enumerate(best_moves):
        target_policy[i][best_move.mask.bit_length() - 1 if best_move else 0] = 0.9

    target_value = th.zeros((batch_size,))
    for i, (n_moves, best_score, reward) in enumerate(
        zip(n_movess, best_scores, rewards)
    ):
        discount = gamma ** ((board.size_square - n_moves) / 2)
        target_value[i] = float(reward) + (1.0 if best_score > 0 else -1.0) * discount

    target_value = target_value.to(Device.XPU)
    target_policy = target_policy.to(Device.XPU)
    max_probs = max_probs.to(Device.XPU)
    loss = compute_loss(
        batched_probs, value, target_policy, target_value, max_probs, step
    )
    loss.backward()
    # TODO: make sense of this function such that actor and critic have independent losses
    actor_optimizer.step()
    critic_optimizer.step()
    # optimizer_scheduler.step()


def train_on_replay(replay_buffer: ReplayBuffer, step: int):
    for _ in range(replay_runs):
        obs, n_movess, best_moves, best_scores, rewards = replay_buffer.sample(
            replay_batch_size
        )
        batched_probs, batched_probs_clone, max_probs, value = run_model(obs)
        train(
            n_movess,
            best_moves,
            best_scores,
            rewards,
            batched_probs,
            value,
            max_probs,
            step,
            replay_batch_size,
        )


def get_moves_from_probs(batched_probs):
    return batched_probs.argmax(dim=1)


def generate_unoccupied_from_distrib(board_size, batched_probs_clone, boards):
    batched_probs_clone /= batched_probs_clone.sum(dim=-1, keepdims=True)
    occupied_for_each_env = (board.occupied for board in boards)
    for i, occ in enumerate(occupied_for_each_env):
        for mask in generate_masks(occ):
            print(
                f"setting {i} to 0, mask {mask.bit_length() - 1}",
                Move(mask, size=board_size).get_coord(),
            )
            batched_probs_clone[i][mask.bit_length() - 1] = 0

    distrib = th.distributions.Categorical(batched_probs_clone)
    moves = distrib.sample()
    scores = batched_probs_clone.gather(1, moves.unsqueeze(0)).squeeze()
    actions = th.stack((moves, scores), dim=1)
    # print(moves, scores, actions)


def compute_loss(batched_probs, value, target_policy, target_value, max_probs, step):
    policy_loss = th.nn.functional.cross_entropy(batched_probs, target_policy)

    advantages = target_value - value
    # advantages = (advantages - advantages.mean()) / advantages.std()
    policy_value_loss = (max_probs.log() * advantages.detach()).mean().square()
    policy_value_loss_weight = abs(
        policy_loss.item() / (policy_value_loss.item() + 1e-8)
    )

    value_loss = th.nn.functional.mse_loss(value, target_value)
    value_loss_weight = abs(policy_loss.item() / (value_loss.item() + 1e-8))

    entropy = -th.sum(batched_probs * (batched_probs + 1e-8).log(), dim=-1).mean()
    entropy_weight = abs(policy_loss.item() / (entropy.item() + 1e-8))

    policy_value_loss_queue.append(policy_value_loss.item())
    policy_loss_queue.append(policy_loss.item())
    value_loss_queue.append(value_loss.item())
    entropy_queue.append(entropy.item())

    heuristic_weight = ((total_steps - step) / total_steps) ** 2
    return (
        policy_loss * heuristic_weight
        + policy_value_loss * (1 - heuristic_weight) * policy_value_loss_weight
        + value_loss * value_loss_weight * value_coef
        + entropy * entropy_weight * get_adjusted_entropy(ent_coef, step)
    )


def log_progress(env, step, actor_gradient, critic_gradient):
    actions = np.array(action_queue, dtype=np.float16)
    max_actions = np.array(max_action_queue, dtype=np.float16)
    # print(actions.size, max_actions.size)
    writer.add_histogram(
        "actions/probs",
        actions,
        bins="auto",
        max_bins=100,
    )
    writer.add_histogram(
        "actions/top_probs",
        max_actions,
        bins="auto",
        max_bins=20,
    )
    writer.add_scalar("episode/length", np.mean(env.length_queue), step)
    writer.add_scalar("episode/fps", np.mean(env.time_queue) * n_envs, step)
    # writer.add_scalar("rewards/mean_total", np.mean(env.return_queue), all_env_steps)
    writer.add_scalar("rewards/mean_winner", np.mean(env.winner_queue), step)
    writer.add_scalar("rewards/mean_final", np.mean(env.reward_queue), step)
    writer.add_scalar("losses/policy_loss", np.mean(policy_loss_queue), step)
    writer.add_scalar(
        "losses/policy_value_loss", np.mean(policy_value_loss_queue), step
    )
    writer.add_scalar("losses/value_loss", np.mean(value_loss_queue), step)
    writer.add_scalar("misc/entropy", np.mean(entropy_queue), step)
    writer.add_scalar("misc/ent_coef", get_adjusted_entropy(ent_coef, step), step)
    writer.add_scalar("misc/actor_gradient", actor_gradient, step)
    writer.add_scalar("misc/critic_gradient", critic_gradient, step)
    # writer.add_scalar("misc/lr", get_learning_rate_decay()(step), step)


def collect_worker_responses(boards_chunks: Iterable[tuple[HexBoard]]):
    responses = []
    chunks = 0
    for boards in boards_chunks:
        chunks += 1
        last_chunk_size = 0
        for (_, in_queue, _), board in zip(workers, boards):
            in_queue.put(board.serialize_position())
            last_chunk_size += 1

    for i in range(chunks):
        for k, (_, _, out_queue) in enumerate(workers):
            if i == chunks - 1 and k == last_chunk_size - 1:
                break
            responses.append(out_queue.get(block=True, timeout=None))

    return responses


def worker_loop(board_size, worker_id: int, in_queue: Queue, out_queue: Queue):
    board = HexBoard(size=board_size)
    i = 0
    while True:
        board_bytes = in_queue.get(block=True, timeout=60)
        board.deserialize_position(board_bytes)
        out_queue.put(get_logical_move(board))

        if worker_id == 0 and i % 1000 == 0:
            cache_info = board.distance_missing_cached.cache_info()
            hits = cache_info.hits
            ratio = cache_info.misses / hits
            print(f"Distance Cache: hits={hits}, ratio={ratio}")

        i += 1


def get_logical_move(board: HexBoard) -> tuple[Move, float, int]:
    n_moves = board.size_square - board.unoccupied.bit_count()
    color = board.turn

    best_move: Move | None = None
    best_score = None
    for move in board.legal_moves:
        # score = _get_distance_perf(board)
        score = _get_distance_score(board, color, n_moves)

        if best_move is None or (
            ((color and score > best_score) or (not color and score < best_score))
        ):
            best_move = move
            best_score = score
    return best_move, best_score, n_moves


def _get_distance_perf(board) -> float:
    white_missing = board.get_shortest_missing_distance_perf(True)
    black_missing = board.get_shortest_missing_distance_perf(False)

    if not white_missing:
        return 1
    return np.tanh((black_missing / white_missing) - 1)


def _get_distance_score(board, color, n_moves: int) -> float:
    (
        white_missing,
        white_variants,
    ) = board.get_short_missing_distances(
        True, should_subtract=(not color and n_moves % 2 == 1)
    )  # subtracts distance from white because has 1 stone less on board, on odd moves
    (
        black_missing,
        black_variants,
    ) = board.get_short_missing_distances(False)

    white_score = sum(
        (_weight_distance(board, board.size - k, n_moves) * v)
        for k, v in white_variants.items()
    )
    black_score = sum(
        (_weight_distance(board, board.size - k, n_moves) * v)
        for k, v in black_variants.items()
    )

    if not black_score:
        return 1
    return np.tanh((white_score / black_score) - 1)


def _weight_distance(board, distance, n_moves) -> int:
    max_moves = board.size_square
    if n_moves > max_moves / 2:
        return distance**3 / max_moves
    elif n_moves > max_moves / 4:
        return distance**2 / board.size
    else:
        return distance


if __name__ == "__main__":
    board_size = 7
    board = HexBoard("", size=board_size, use_graph=True)
    writer = SummaryWriter(os.path.join(LOG_PATH, f"custom_tensorboard", f"custom6"))
    workers = []
    for i in range(n_workers):
        in_queue = Queue()
        out_queue = Queue()
        process = Process(
            target=worker_loop, args=(board_size, i, in_queue, out_queue), daemon=True
        )
        process.start()
        workers.append((i, in_queue, out_queue))

    actor_model, critic_model, actor_optimizer, critic_optimizer = prepare_model()
    # actor_optimizer_scheduler = LambdaLR(actor_optimizer, get_learning_rate_decay())
    # critic_optimizer_scheduler = LambdaLR(actor_optimizer, get_learning_rate_decay())
    try:
        run(writer, actor_model, critic_model, actor_optimizer, critic_optimizer)
    finally:
        writer.close()

        if not os.path.exists("custom"):
            os.mkdir("custom")
        th.save(actor_model.state_dict(), f"custom/custom-model-black.v0")
        th.save(critic_model.state_dict(), f"custom/custom-model-black.v0")
        th.save(actor_optimizer.state_dict(), f"custom/custom-optimizer-black.v0")
        th.save(critic_optimizer.state_dict(), f"custom/custom-optimizer-black.v0")
