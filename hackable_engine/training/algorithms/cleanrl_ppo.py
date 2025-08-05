from argparse import ArgumentParser
import asyncio
import os
from itertools import chain

import numpy as np
import torch as th
from torch.distributions import Categorical
from torch.nn import Linear
from torch.optim import  AdamW
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import MultiprocessAsyncEnv
from hackable_engine.training.envs.multiprocess_vector_env.safe_sync_vector_env import SafeSyncVectorEnv
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_gmm import GraphGMM
from hackable_engine.training.utils.device import Device
import torch.nn.functional as F

LOG_PATH = "./hackable_engine/training/logs/"
# OPPONENT_VARIANTS = ["gmm1", "gmm2", "sg", "gin", "gine", "rgcn", "gat", "gatformer", "gmmformer"]
OPPONENT_VARIANTS = ["gmm110", "gmm220", "gmm330", "gmm440", "gmm550", "gmm511", "gmm522", "gmm533", "gmm544"]
BOARD_SIZE = 13

N_WORKERS = 8
N_ENVS = 128
N_STEPS = 32
TOTAL_TIMESTEPS = 1_000_000
N_EPOCHS = 8

LEARNING_RATE = 3e-4
MAX_GRAD_NORM = 10_000.0
BATCH_SIZE = 64
expected_episode_steps = 64
GAMMA = (expected_episode_steps - 1) / expected_episode_steps
CLIP_RANGE = 0.5
GAE_LAMBDA = 0.9
ENT_COEF = 0.001
ACTION_TEMPERATURE = 0.005
VF_COEF = 0.5
TARGET_KL = None

NORMALIZE_ADVANTAGES = True
CLIP_VLOSS = True

GNN_SHAPE = (64, 96, 128, 160, 192, 224)
MLP_SHAPE = (256,)
DROPOUTS = 0.25

WEIGHT_DECAY = 3e-4
ADAMW_BETAS = (0.9, 0.999)


class Agent(th.nn.Module):

    def __init__(self, model, device: Device):
        super().__init__()
        self.model = model
        self.device = device

    def get_value(self, data):
        with th.amp.autocast("xpu", dtype=th.bfloat16, enabled=True):
            _, value = self.model(data)
        return value

    def get_action_and_value(self, data, action=None):
        with th.amp.autocast("xpu", dtype=th.bfloat16, enabled=True):
            logits, value = self.model(data)

        logits = logits / ACTION_TEMPERATURE

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
            # action = self._get_softmax_action(logits, 0.001)

        logprob = probs.log_prob(action)
        return action.unsqueeze(1), logprob.unsqueeze(1), probs.entropy(), value

    def _get_softmax_action(self, logits: th.Tensor) -> int:
        probabilities = F.softmax(logits, dim=0)
        action = th.multinomial(probabilities, num_samples=1).item()
        return action


def run(version: int, load_version: int, env_class, device, color: bool, is_transfer: bool = False):
    base_path = init_directory(env_class.__name__, version)
    writer = SummaryWriter(
        os.path.join(LOG_PATH, f"ppo_{BOARD_SIZE}_{int(color)}", f"{env_class.__name__}_{version}")
    )
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )
    envs = MultiprocessAsyncEnv(
        lambda process_id, num_envs, color_: EpisodeStats(
            SafeSyncVectorEnv(
                [
                    lambda: env_class(
                        color=color_,
                        process_id=process_id,
                        env_id=env_id,
                        num_processes=N_WORKERS,
                        num_envs=num_envs,
                        models=OPPONENT_VARIANTS,
                    )
                    for env_id in range(num_envs)
                ],
                copy=False,
            ),
            is_multiprocessed=True,
        ),
        N_WORKERS,
        int(N_ENVS // N_WORKERS),
        action_shape=(1,),
        color=args.color,
    )

    board = HexBoard("", size=BOARD_SIZE)
    model_kwargs = dict(
        node_count=board.size_square,
        node_features=9,
        output_size=1,
        batch_size=BATCH_SIZE,
        dropouts=DROPOUTS,
        num_envs=N_ENVS,
        gnn_shape=GNN_SHAPE,
        # gnn_heads=GNN_HEADS,
        mlp_shape=MLP_SHAPE,
        edge_index=board.edge_index,
        # edge_types=board.edge_types,
        # edge_types=board.edge_types_rgcn,
        pseudo_coordinates=board.pseudo_coordinates,
        # kernel_size=3,
        # gmm_layers=2,
    )
    model = GraphGMM(**model_kwargs, is_actor_critic=True)
    model = model.to(device)
    model.train(mode=True)

    # optimizer = Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)
    optimizer_params = [
        {
            "params": [param for layer in chain(model.gnn, model.residuals) for param in layer.parameters()],
            # "params": [param for layer in chain(model.gnn, model.residuals, model.norms) for param in layer.parameters()],
            "lr": 0.0 if is_transfer else 0.0,#LEARNING_RATE,
        },
        {
            "params": [param for layer in model.policy_head for param in layer.parameters()],
            "lr": 0.0 if is_transfer else 0.0,#LEARNING_RATE,
        },
        {
            "params": [param for layer in model.value_head for param in layer.parameters()],
            "lr": LEARNING_RATE * 1.5,
        },
    ]
    optimizer = AdamW(optimizer_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=ADAMW_BETAS)
    if is_transfer:
        load_dqn_weights(model, color, load_version)
    elif load_version is not None:
        load_model_if_available(base_path, env_class.__name__, load_version, model, color, optimizer)
    model = th.compile(model)
    agent = Agent(model, device).to(device)

    # ALGO Logic: Storage setup
    obs = th.zeros((N_STEPS, N_ENVS) + envs.single_observation_space.shape).to(device)
    actions = th.zeros((N_STEPS, N_ENVS, board.size_square)).to(device)
    logprobs = th.zeros((N_STEPS, N_ENVS, board.size_square)).to(device)
    rewards = th.zeros((N_STEPS, N_ENVS)).to(device)
    dones = th.zeros((N_STEPS, N_ENVS)).to(device)
    values = th.zeros((N_STEPS, N_ENVS)).to(device)

    try:
        asyncio.run(
            train(envs, agent, optimizer, writer, device, obs, actions, values, rewards, dones, logprobs, is_transfer)
        )
    finally:
        save_models(
            base_path,
            env_class.__name__,
            version,
            agent,
            optimizer,
            color,
        )

        writer.close()


def load_model_if_available(base_path, env_name, version, model, color, optimizer):
    print(base_path)
    if os.path.exists(base_path):
        print("loading pre-trained weights...")

        color_txt = "white" if color else "black"
        pretrained_weights = th.load(f"{base_path}/{env_name}-ppo-{color_txt}.v{version}", weights_only=True)
        pretrained_weights = {key.replace("_orig_mod.", ""): value for key, value in pretrained_weights.items()}

        model.load_state_dict(pretrained_weights)
        optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-ppo-optimizer-{color_txt}.v{version}", weights_only=True
            )
        )


def load_dqn_weights(model, color, version: int):
    print("loading dqn weights for value head training")
    model_state_dict = model.state_dict()
    color_txt = "white" if color else "black"
    pretrained_weights = th.load(f"dqn/{BOARD_SIZE}/dqn-model-{color_txt}.v{version}")
    # print(list(pretrained_weights.keys()))

    if "_orig_mod" in pretrained_weights:
        pretrained_weights = pretrained_weights["_orig_mod"]

    pretrained_weights["policy_head.0.weight"] = pretrained_weights["mlp.0.weight"]
    pretrained_weights["policy_head.2.weight"] = pretrained_weights["mlp.1.weight"]
    pretrained_weights["policy_head.0.bias"] = pretrained_weights["mlp.0.bias"]
    pretrained_weights["policy_head.2.bias"] = pretrained_weights["mlp.1.bias"]

    new_weights = {k: v for k, v in pretrained_weights.items() if k in model_state_dict}
    print(list(new_weights.keys()))
    model_state_dict.update(new_weights)
    model.load_state_dict(model_state_dict)


def load_optimizer_weights(optimizer):
    optimizer_state_dict = optimizer.state_dict()
    pretrained_weights = th.load("dqn/11/dqn-optimizer-black.v1")

    optimizer_state_dict["state"].update(pretrained_weights["state"])
    optimizer.load_state_dict(optimizer_state_dict)


def save_models(new_path, env_name, version, agent, optimizer, color):
    color_txt = "white" if color else "black"
    th.save(agent.model.state_dict(), f"{new_path}/{env_name}-ppo-{color_txt}.v{version}")
    th.save(
        optimizer.state_dict(), f"{new_path}/{env_name}-ppo-optimizer-{color_txt}.v{version}"
    )


def init_directory(env_name, version) -> tuple[str, str]:
    base_path = f"./ppo-{env_name}"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    try:
        with open(f"{base_path}/hyperparams-{env_name}-v{version}.log", "w") as f:
            for param in [
                N_ENVS,
                TOTAL_TIMESTEPS,
                LEARNING_RATE,
                MAX_GRAD_NORM,
                N_STEPS,
                BATCH_SIZE,
                GAMMA,
                CLIP_RANGE,
                GAE_LAMBDA,
                ENT_COEF,
            ]:
                name = get_variable_name(param)
                if name:
                    f.write(f"{name}={param}\n")
    except ValueError:
        print("hyperparams file not created")
    return base_path


def get_variable_name(v):
    for pair in locals().copy():
        try:
            name, val = pair
            if v is val:
                return name
        except ValueError:
            continue
    raise ValueError("variable not found")


async def train(
    env: MultiprocessAsyncEnv,
    agent: Agent,
    optimizer,
    writer,
    device,
    obs,
    actions,
    values,
    rewards,
    dones,
    logprobs,
    is_transfer,
):
    global_step = 0
    next_obs, _, _ = env.reset(seed=1)
    next_obs = th.from_numpy(np.stack(next_obs, 0)).to(Device.XPU)
    next_done = th.zeros(N_ENVS).to(device)
    num_iterations = TOTAL_TIMESTEPS // N_STEPS

    for iteration in range(1, num_iterations + 1):
        # # Annealing the rate if instructed to do so.
        # if args.anneal_lr:
        #     frac = 1.0 - (iteration - 1.0) / args.num_iterations
        #     lrnow = frac * LEARNING_RATE
        #     optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, N_STEPS):
            global_step += N_ENVS
            obs[step] = next_obs.squeeze(1)
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with th.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, _, _, _ = await env.step(  # takes 4 to 10 times longer than model above
                action.cpu().numpy()
            )

            # next_done = np.logical_or(terminations, truncations)
            rewards[step] = th.tensor(reward).to(device).view(-1)
            next_obs, next_done = th.Tensor(next_obs).to(device), th.Tensor(
                terminations
            ).to(device)

        # bootstrap value if not done
        with th.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = th.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(N_STEPS)):
                if t == N_STEPS - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + GAMMA * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + env.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(N_STEPS * N_ENVS)
        clipfracs = []
        v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl = train_epochs(agent, optimizer, b_inds, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, clipfracs, is_transfer)

        gnn_gradients = []
        for param in [param for layer in agent.model.gnn.modules() for param in layer.parameters()]:
            if param.grad is not None:
                gnn_gradients.append(param.grad.view(-1))
        gnn_gradients = th.cat(gnn_gradients)
        gnn_gradient = gnn_gradients.norm()

        mlp_gradients = []
        for param in [param for layer in chain(agent.model.policy_head.modules(), agent.model.value_head.modules()) for param in layer.parameters() if isinstance(layer, Linear)]:
            if param.grad is not None:
                mlp_gradients.append(param.grad.view(-1))
        mlp_gradients = th.cat(mlp_gradients)
        mlp_gradient = mlp_gradients.norm()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        try:
            env_progress_data: EnvProgressData = env.get_progress_data()
        except ZeroDivisionError:
            print("Zero division error")  # FIXME
            return
        # writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        # writer.add_histogram(
        #     "charts/actions",
        #     np.array(env.action_queue),
        #     bins="auto",
        #     max_bins=100,
        # )
        print("logging...")
        writer.add_scalar("episode/length", env_progress_data.length_mean, iteration)
        writer.add_scalar("episode/win_length", env_progress_data.win_length_mean, iteration)
        writer.add_scalar("episode/win_length_std", env_progress_data.win_length_std, iteration)
        writer.add_scalar("episode/loss_length", env_progress_data.loss_length_mean, iteration)
        writer.add_scalar("episode/loss_length_std", env_progress_data.loss_length_std, iteration)
        writer.add_scalar("episode/fps", env_progress_data.time_mean * N_ENVS, iteration)
        writer.add_scalar("rewards/mean_total", env_progress_data.return_mean, iteration)
        writer.add_scalar("rewards/mean_winner", env_progress_data.winner_mean, iteration)
        writer.add_scalar("rewards/mean_final", env_progress_data.reward_mean, iteration)
        writer.add_scalar("gradients/gnn", gnn_gradient, iteration)
        writer.add_scalar("gradients/mlp", mlp_gradient, iteration)
        # % of games that finished without any illegal moves
        writer.add_scalar("rewards/%_legal", env_progress_data.legal_mean, iteration)
        writer.add_scalar("losses/advantages", lastgaelam.mean().item(), iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/entropy", entropy_loss.item(), iteration)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iteration)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), iteration)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), iteration)
        writer.add_scalar("losses/explained_variance", explained_var, iteration)

        # print(f"mean_reward={mean_reward}, explained_var={explained_var}, approx_kl={approx_kl}, value_loss={v_loss.item()}, policy_loss={pg_loss.item()}, entropy_loss={entropy_loss.item()}")

@th.compile
def train_epochs(agent, optimizer, b_inds, b_obs, b_actions, b_logprobs, b_advantages, b_returns, b_values, clipfracs, is_transfer):
    for epoch in range(N_EPOCHS):
        np.random.shuffle(b_inds)
        for start in range(0, N_STEPS * N_ENVS, BATCH_SIZE):
            end = start + BATCH_SIZE
            mb_inds = b_inds[start:end]

            _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
            logratio = newlogprob - b_logprobs[mb_inds]
            ratio = logratio.exp()

            with th.no_grad():
                # calculate approx_kl http://joschu.net/blog/kl-approx.html
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfracs += [((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()]

            mb_advantages = b_advantages[mb_inds]
            if NORMALIZE_ADVANTAGES:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            # Policy loss
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * th.clamp(ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE)
            pg_loss = th.max(pg_loss1, pg_loss2).mean()

            # Value loss
            newvalue = newvalue.view(-1)
            if CLIP_VLOSS:
                v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                v_clipped = b_values[mb_inds] + th.clamp(
                    newvalue - b_values[mb_inds],
                    -CLIP_RANGE,
                    CLIP_RANGE,
                )
                v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                v_loss_max = th.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

            entropy_loss = entropy.mean()
            if is_transfer:
                loss = v_loss
            else:
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

            optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(agent.model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

    return v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-c",
        "--color",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-v",
        "--version",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-l",
        "--load-version",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-t",
        "--is-transfer",
        action="store_true",
    )
    args = parser.parse_args()
    env_class = Logit13GraphEnv if BOARD_SIZE == 13 else Logit11GraphEnv
    run(args.version, args.load_version, env_class, Device.XPU, bool(args.color), args.is_transfer)
