# -*- coding: utf-8 -*-

RESET_CHARTS = True
N_ENV_WORKERS = 10
"""Not really a hyperparam, but impacts the choice of others."""

N_ENVS = 64 * N_ENV_WORKERS
TOTAL_TIMESTEPS: int = 2**26
LEARNING_RATE: float = 3e-3  #lambda p: 0 if p > 0.998 else 3e-5

# slowly decline to a point
# LEARNING_RATE = lambda p: max(1e-3 * p**2, 1e-5)

# start small, grow rapidly and then slowly decline
# LEARNING_RATE = lambda p: 1e-6 if p > 0.95 else min(1e-6 * (2-p)**9, 5e-4) if p > 0.9 else max(5e-4 * p**4, 1e-5)

# LEARNING_RATE = lambda p: 1e-6 + 1e-4 * (1 + np.sin(p*100))
# LEARNING_RATE = lambda p: 0 if p > 0.998 else max(3e-4 * p**2, 1e-5)

SGD_MOMENTUM = (0.9, 0.9)  # initial and used in subsequent training values
"""Only used when SGD optimizer is chosen for a policy."""

SGD_DAMPENING = (0.0, 0.0)  # initial and used in subsequent training values
"""Only used when SGD optimizer is chosen for a policy."""

ADAMW_WEIGHT_DECAY = (1e-6, 1e-6)  # initial and used in subsequent training values
"""Only used when AdamW/Adam optimizer is chosen for a policy."""

N_EPOCHS: int = 128
N_STEPS: int = 2**5
"""Batch size per env, ie. will update policy every `N_STEPS` iterations, total batch size is this times `N_ENVS`."""

# BATCH_SIZE: int = int(N_ENVS // N_ENV_WORKERS * N_ENV_WORKERS * N_STEPS / 2**0)
BATCH_SIZE = 32#2**14
"""So called mini-batch, size taken into GPU at once, recommended to be a factor of (`N_STEPS * N_ENVS`)."""

CLIP_RANGE: float = 0.3
"""
Clip value for PPO loss
0.1 to 0.3 according to many sources, but I used even 0.9 with beneficial results.
"""

MAX_GRAD_NORM: float = 2.0
"""
Gradient is clipped to avoid values exploding to infinity. Actually seems in SB3 it's multiplied by a value < 1.
clip_coef_clamped = torch.clamp(`MAX_GRAD_NORM` / (total_norm + 1e-6), max=1.0)
torch._foreach_mul_(grads, clip_coef_clamped)
"""

STD_INIT: float = 0.5
"""An initial standard deviation for the probability distribution of an action taken by the actor."""

GAMMA: float = 0.99  # 0.9975 for 9x9
"""
Discount factor for the past actions in an episode.
Has to be <1, in practice when very close to 1 then is more difficult for the model to learn efficiently.
One way to choose it: 1 / (1 - GAMMA) = number of steps to finish the episode. G = (N - 1)/N
"""

GAE_LAMBDA: float = 0.90
"""0.9 to 0.99, controls the trade-off between bias and variance, i.e. underfitting and overfitting."""

ENT_COEF: float = 0.01
"""
Entropy: 0 to 0.01 https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
Proved very useful, in practice possibly even set to high values like 0.3.
"""

### CLEANRL ###

NORMALIZE_ADVANTAGES: bool = True
""""""

CLIP_VLOSS: bool = True
"""Toggles whether or not to use a clipped loss for the value function."""

VF_COEF: float = 0.5
"""Coefficient of the value function."""

# OFF POLICY #

BUFFER_SIZE: int = 32_000_000  # more-less limited to 2**26 by the amount of RAM
"""The replay memory buffer size."""

Q_LEARNING_RATE = 1.5e-3
ALPHA_LEARNING_RATE = 1e-5 # Q_LEARNING_RATE
"""Slowing down entropy decrease, hoping for longer exploration as opposed to exploitation"""

LEARNING_STARTS: int = int(BUFFER_SIZE * 0.005)
"""Number of random moves before the agent starts choosing them."""

ENTROPY_AUTOTUNE: bool = True
MIN_ENTROPY_ALPHA: float = 0.001
ENTROPY_ALPHA: float = 0.1
"""Entropy regularization coefficient."""

TAU: float = 0.01  # default was 0.005
"""Target smoothing coefficient."""

POLICY_FREQUENCY: int = 2
Q_FREQUENCY: int = 2
"""The frequency of training policy (delayed)."""

TARGET_NETWORK_FREQUENCY: int = 1  # Denis Yarats' implementation delays this by 2.
"""The frequency of updates for the target networks."""

POLICY_NOISE: float = 0.2
"""The scale of policy noise."""

EXPLORATION_NOISE: float = 0.1
"""The scale of exploration noise."""

NOISE_CLIP: float = 0.5
"""Noise clip parameter of the Target Policy Smoothing Regularization."""
