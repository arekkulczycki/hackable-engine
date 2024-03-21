# -*- coding: utf-8 -*-
import torch as th

from hackable_engine.training.network.alphazero_features_extractor import AlphaZeroFeaturesExtractor
from hackable_engine.training.envs.hex.raw_9_binary_env import Raw9BinaryEnv
from hackable_engine.training.envs.hex.raw_9_channelled_env import Raw9ChannelledEnv
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env
from hackable_engine.training.network.hex_cnn_features_extractor import HexCnnFeaturesExtractor
from hackable_engine.training.hyperparams import *

cnn_base = dict(
    policy="CnnPolicy",
    # optimizer_class=th.optim.AdamW,
    # optimizer_kwargs=dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
    optimizer_class=th.optim.SGD,
    optimizer_kwargs=dict(momentum=SGD_MOMENTUM[0], dampening=SGD_DAMPENING[0], nesterov=True),
    features_extractor_class=HexCnnFeaturesExtractor,
    features_extractor_kwargs=dict(board_size=7, n_filters=(32,), kernel_sizes=(3,), should_normalize=False, activation_fn=th.nn.Tanh),
    should_preprocess_obs=False,
    net_arch=[64, 64],
    # use_expln=True,
    # activation_fn=th.nn.Sigmoid,
)

policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "hex9cnnA": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(256,), kernel_sizes=(5,), strides=(2,), resnet_layers=2, activation_fn=th.nn.Tanh
        ),
        "activation_fn": th.nn.ReLU,  # for the fully connected layer, another interesting is th.nn.Tanhshrink
        "net_arch": [8, 8],
    },
    "hex9cnnB": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(256,), kernel_sizes=(5,), strides=(2,), activation_fn=th.nn.Tanh
        ),
        "activation_fn": th.nn.LeakyReLU,
        "net_arch": [128, 128],
    },
    "hex9cnnC": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(256,), kernel_sizes=(5,), strides=(2,)
        ),
        "activation_fn": th.nn.Tanhshrink,
        "net_arch": [128, 128],
    },
    "hex9cnnD": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(64, 64), kernel_sizes=(3, 2), strides=(2, 2), activation_fn=th.nn.Tanh
        ),
        "optimizer_class": th.optim.AdamW,
        "optimizer_kwargs": dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
        "net_arch": [8, 8],
    },
    "hex9cnnE": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(256, 64), kernel_sizes=(5, 3), strides=(2, 1)
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [8, 8],
    },
    "hex9cnnF": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(64, 32, 32), kernel_sizes=(3, 3, 3)),
        "activation_fn": th.nn.ReLU,
        "net_arch": [16, 16],
    },
    "hex9raw": {
        "env_class": Raw9BinaryEnv,
        "net_arch": [162, 162],
        # "optimizer_class": th.optim.AdamW,
    },
    "hex9az": {
        "env_class": Raw9ChannelledEnv,
        "features_extractor_class": AlphaZeroFeaturesExtractor,
        "features_extractor_kwargs": dict(num_actions=1),
    },
}
