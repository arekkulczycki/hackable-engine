import asyncio
import os

import numpy as np
import torch as th
from torch.distributions import Categorical
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard as HexBoard
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.batch_inference_vector_env import BatchInferenceVectorEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import MultiprocessAsyncEnv
from hackable_engine.training.envs.multiprocess_vector_env.util import EnvProgressData
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats
from hackable_engine.training.models.graph_gmm_ac import GraphGmmAC
from hackable_engine.training.utils.device import Device

LOG_PATH = "./hackable_engine/training/logs/"
OPPONENT_VARIANTS = ["gmm1", "gmm2", "sg", "gin", "gine", "rgcn", "gat", "gatformer", "gmmformer"]
BOARD_SIZE = 11

N_WORKERS = 8
N_ENVS = 256  # TODO: for simplicity for now N_ENVS==BATCH_SIZE, otherwise I need to change the model forward function
N_STEPS = 16
TOTAL_TIMESTEPS = 1_000_000
N_EPOCHS = 8

LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 0.5
BATCH_SIZE = 256
GAMMA = 0.99
CLIP_RANGE = 0.2
GAE_LAMBDA = 0.95
ENT_COEF = 0.01
VF_COEF = 0.5
TARGET_KL = None

NORMALIZE_ADVANTAGES = False
CLIP_VLOSS = True

GNN_HIDDEN = 72


class Agent(th.nn.Module):

    def __init__(self, model: GraphGmmAC, device: Device):
        super().__init__()
        self.model = model
        self.device = device

        self.hidden_dim = model.trunk.lstm.hidden_size

    def init_hidden(self, batch_size):
        h = th.zeros(1, batch_size, self.hidden_dim, device=self.device)
        c = th.zeros(1, batch_size, self.hidden_dim, device=self.device)
        return (h, c)

    def get_value(self, data, hidden):
        _, value, _ = self.model(data, hidden)
        return value

    def get_action_and_value(self, data, hidden, action=None):
        """
        data: PyG Batch
        hidden: LSTM hidden state
        action: optional (for PPO update)
        """

        logits, value, hidden = self.model(data, hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value, hidden


def run(version, env_class, device, color):
    base_path, new_path = init_directory(env_class.__name__, version)
    writer = SummaryWriter(
        os.path.join(LOG_PATH, f"{env_class.__name__}_tensorboard", f"{env_class.__name__}_{version}")
    )
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )
    envs = MultiprocessAsyncEnv(
        lambda process_id, num_envs, color_: EpisodeStats(
            BatchInferenceVectorEnv(
                [
                    lambda models: env_class(
                        color=color_,
                        process_id=process_id,
                        env_id=env_id,
                        num_processes=N_WORKERS,
                        num_envs=num_envs,
                        models=models,
                    )
                    for env_id in range(num_envs)
                ],
                copy=False,
                color=color_,
                model_variants=OPPONENT_VARIANTS,
            ),
            is_multiprocessed=True,
        ),
        N_WORKERS,
        int(N_ENVS // N_WORKERS),
        action_shape=(1,),
        color=color,
    )

    board = HexBoard("", size=BOARD_SIZE)
    model = GraphGmmAC(node_dim=9, gnn_hidden=GNN_HIDDEN, trunk_hidden=256, num_actions=BOARD_SIZE**2, edge_index=board.edge_index, pseudo_coord=board.pseudo_coordinates, node_count=BOARD_SIZE**2, batch_size=BATCH_SIZE)
    # model = th.compile(model)
    agent = Agent(model, device).to(device)

    optimizer = Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = th.zeros((N_STEPS, N_ENVS) + envs.single_observation_space.shape).to(device)
    actions = th.zeros((N_STEPS, N_ENVS) + envs.single_action_space.shape).to(device)
    logprobs = th.zeros((N_STEPS, N_ENVS)).to(device)
    rewards = th.zeros((N_STEPS, N_ENVS)).to(device)
    dones = th.zeros((N_STEPS, N_ENVS)).to(device)
    values = th.zeros((N_STEPS, N_ENVS)).to(device)
    hidden_h = th.zeros((N_STEPS, 1, BATCH_SIZE, model.trunk.lstm.hidden_size)).to(device)
    hidden_c = th.zeros((N_STEPS, 1, BATCH_SIZE, model.trunk.lstm.hidden_size)).to(device)

    # load_model_if_available(base_path, env_class.__name__, version, agent, optimizer)

    try:
        # context = (
        #     th.amp.autocast("xpu", enabled=True, dtype=th.float16)
        #     if device == Device.XPU
        #     else nullcontext()
        # )
        # with context:
        asyncio.run(train(
            envs,
            agent,
            optimizer,
            writer,
            device,
            obs,
            actions,
            values,
            rewards,
            dones,
            hidden_h,
            hidden_c,
            logprobs,
        ))
    finally:
        save_models(
            new_path,
            env_class.__name__,
            version,
            agent,
            optimizer,
        )

        # envs.close()
        writer.close()


def load_model_if_available(base_path, env_name, version, agent, optimizer):
    if os.path.exists(base_path):
        print("loading pre-trained weights...")
        agent.load_state_dict(
            th.load(f"{base_path}/{env_name}-ppo-agent.v{version}", weights_only=True)
        )
        optimizer.load_state_dict(
            th.load(
                f"{base_path}/{env_name}-ppo-optimizer.v{version}", weights_only=True
            )
        )


def save_models(new_path, env_name, version, agent, optimizer):
    th.save(agent.state_dict(), f"{new_path}/{env_name}-ppo-agent.v{version + 1}")
    th.save(
        optimizer.state_dict(), f"{new_path}/{env_name}-ppo-optimizer.v{version + 1}"
    )


def init_directory(env_name, version) -> tuple[str, str]:
    base_path = f"./{env_name}-ppo.v{version}"
    new_path = f"./{env_name}-ppo.v{version + 1}"
    if not os.path.exists(new_path):
        os.mkdir(new_path)
        try:
            with open(f"{new_path}/hyperparams.log", "w") as f:
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
    hidden_h,
    hidden_c,
    logprobs,
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

        hidden = agent.init_hidden(BATCH_SIZE)

        for step in range(0, N_STEPS):
            global_step += N_ENVS
            obs[step] = next_obs.squeeze(1)
            dones[step] = next_done
            hidden_h[step] = hidden[0]
            hidden_c[step] = hidden[1]

            # ALGO LOGIC: action logic
            with th.no_grad():
                action, logprob, _, value, hidden = agent.get_action_and_value(next_obs, hidden)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, _, _, _ = await env.step(
                action.cpu().numpy()
            )

            # reset lstm memory for terminated envs
            mask = th.tensor(terminations, device=device, dtype=th.float32)
            mask = (1.0 - mask).view(1, -1, 1)
            hidden = (hidden[0] * mask, hidden[1] * mask)

            # next_done = np.logical_or(terminations, truncations)
            rewards[step] = th.tensor(reward).to(device).view(-1)
            next_obs, next_done = th.Tensor(next_obs).to(device), th.Tensor(
                terminations
            ).to(device)

            # if "final_info" in infos:
            #     for info in infos["final_info"]:
            #         if info and "episode" in info:
            #             print(
            #                 f"global_step={global_step}, episodic_return={info['episode']['r']}"
            #             )
            #             writer.add_scalar(
            #                 "charts/episodic_return", info["episode"]["r"], global_step
            #             )
            #             writer.add_scalar(
            #                 "charts/episodic_length", info["episode"]["l"], global_step
            #             )

        # bootstrap value if not done
        with th.no_grad():
            next_value = agent.get_value(next_obs, hidden).reshape(1, -1)
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
        b_actions = actions.reshape((-1,) + env.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_hidden_h = hidden_h.transpose(1, 2).reshape(1, N_STEPS * N_ENVS, 256)  # model.trunk.lstm.hidden_size
        b_hidden_c = hidden_c.transpose(1, 2).reshape(1, N_STEPS * N_ENVS, 256)  # model.trunk.lstm.hidden_size

        # Optimizing the policy and value network
        b_inds = np.arange(N_STEPS * N_ENVS)
        clipfracs = []
        for epoch in range(N_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, N_STEPS * N_ENVS, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], (b_hidden_h[:, mb_inds, :], b_hidden_c[:, mb_inds, :]), b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with th.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > CLIP_RANGE).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if NORMALIZE_ADVANTAGES:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * th.clamp(
                    ratio, 1 - CLIP_RANGE, 1 + CLIP_RANGE
                )
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
                loss = pg_loss - ENT_COEF * entropy_loss + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                optimizer.step()

            # if TARGET_KL is not None and approx_kl > TARGET_KL:
            #     break

        gnn_gradients = []
        for param in [param for layer in agent.model.gnn.modules() for param in layer.parameters()]:
            if param.grad is not None:
                gnn_gradients.append(param.grad.view(-1))
        gnn_gradients = th.cat(gnn_gradients)
        gnn_gradient = gnn_gradients.norm()

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
        # writer.add_scalar("gradients/mlp", mlp_gradient, episode)
        # % of games that finished without any illegal moves
        writer.add_scalar("rewards/%_legal", env_progress_data.legal_mean, iteration)
        writer.add_scalar("losses/value_loss", v_loss.item(), iteration)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), iteration)
        writer.add_scalar("losses/entropy", entropy_loss.item(), iteration)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), iteration)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), iteration)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), iteration)
        writer.add_scalar("losses/explained_variance", explained_var, iteration)

        # print(f"mean_reward={mean_reward}, explained_var={explained_var}, approx_kl={approx_kl}, value_loss={v_loss.item()}, policy_loss={pg_loss.item()}, entropy_loss={entropy_loss.item()}")


if __name__ == "__main__":
    env_class = Logit13GraphEnv if BOARD_SIZE == 13 else Logit11GraphEnv
    run(1, env_class, Device.XPU, False)
