RESET_CHARTS = True
N_ENV_WORKERS = 10
"""Not really a hyperparam, but impacts the choice of others."""

N_ENVS = 220
TOTAL_TIMESTEPS = int(2**24)
LEARNING_RATE = 3e-5  # lambda p: max(1e-4 * p**3, 1e-6)
N_EPOCHS = 220
N_STEPS = 2**5
"""Batch size per env, ie. will update policy every `N_STEPS` iterations, total batch size is this times `N_ENVS`."""

BATCH_SIZE = int(N_ENVS // N_ENV_WORKERS * N_ENV_WORKERS * N_STEPS / 2**0)
"""So called mini-batch, size taken into GPU at once, recommended to be a factor of (`N_STEPS * N_ENVS`)."""

CLIP_RANGE = 0.3
"""
Clip value for PPO loss: see the equation in the intro for more context.
0.1 to 0.3 according to many sources, but I used even 0.9 with beneficial results.
"""

MAX_GRAD_NORM = 0.5
"""
Gradient is clipped to avoid values exploding to infinity. Actually seems in SB3 it's multiplied by a value < 1.
clip_coef_clamped = torch.clamp(`MAX_GRAD_NORM` / (total_norm + 1e-6), max=1.0)
torch._foreach_mul_(grads, clip_coef_clamped)
"""

STD_INIT = 1.0
"""An initial standard deviation for the probability distribution of an action taken by the actor."""

GAMMA = 0.995
"""
Discount factor for the past actions in an episode.
Has to be <1, in practice when very close to 1 then is more difficult for the model to learn efficiently.
One way to choose it: 1 / (1 - GAMMA) = number of steps to finish the episode. G = (N - 1)/N
"""

GAE_LAMBDA = 0.99
"""0.95 to 0.99, controls the trade-off between bias and variance, i.e. underfitting and overfitting."""

ENT_COEF = 0.001
"""
Entropy: 0 to 0.01 https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
Proved very useful, in practice possibly even set to high values like 0.3.
"""

SGD_MOMENTUM = (0.66, 0.66)  # initial and used in subsequent training values
"""Only used when SGD optimizer is chosen for a policy."""

SGD_DAMPENING = (0.0, 0.0)  # initial and used in subsequent training values
"""Only used when SGD optimizer is chosen for a policy."""

ADAMW_WEIGHT_DECAY = (1e-8, 1e-8)  # initial and used in subsequent training values
"""Only used when AdamW/Adam optimizer is chosen for a policy."""
