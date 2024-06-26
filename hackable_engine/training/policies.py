# -*- coding: utf-8 -*-
import torch as th

from hackable_engine.training.network.alphazero_features_extractor import (
    AlphaZeroFeaturesExtractor,
)
from hackable_engine.training.envs.hex.raw_9_binary_env import Raw9BinaryEnv
from hackable_engine.training.envs.hex.raw_9_channelled_env import Raw9ChannelledEnv
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env
from hackable_engine.training.network.hex_cnn_features_extractor import (
    HexCnnFeaturesExtractor,
)
from hackable_engine.training.hyperparams import *
from hackable_engine.training.network.hex_resnet_features_extractor import HexResnetFeaturesExtractor
from hackable_engine.training.network.hex_resnet_raw_features_extractor import HexResnetRawFeaturesExtractor

cnn_base = dict(
    policy="CnnPolicy",
    # optimizer_class=th.optim.AdamW,
    # optimizer_kwargs=dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
    optimizer_class=th.optim.SGD,
    optimizer_kwargs=dict(
        momentum=SGD_MOMENTUM[0], dampening=SGD_DAMPENING[0], nesterov=True
    ),
    share_features_extractor=False,
    features_extractor_class=HexCnnFeaturesExtractor,
    features_extractor_kwargs=dict(
        board_size=7,
        n_filters=(32,),
        kernel_sizes=(3,),
        should_normalize=False,
        activation_fn=th.nn.Tanh,
    ),
    should_preprocess_obs=False,
    net_arch=[64, 64],
    ortho_init=False,
    log_std_init=th.log(th.tensor(0.3)),
    # use_expln=True,
    # activation_fn=th.nn.Sigmoid,
)

policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "hex9raw": {
        "policy": "MlpPolicy",
        "env_class": Raw9BinaryEnv,
        "net_arch": [162, 162],
        # "optimizer_class": th.optim.AdamW,
        # "optimizer_kwargs": dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
        "activation_fn": th.nn.Tanhshrink,
        "ortho_init": False,
        "log_std_init": th.log(th.tensor(STD_INIT)),
    },
    "hex9cnn0": {
        **cnn_base,
        "env_class": Raw9BinaryEnv,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(128,),
            kernel_sizes=(3,),
            strides=(1,),
            should_normalize=False,
            learning_rate=4 * LEARNING_RATE,
            activation_fn=th.nn.Tanh,
        ),
        "activation_fn": th.nn.Tanhshrink,
        "net_arch": [128, 128],
        # "optimizer_class": th.optim.AdamW,
        # "optimizer_kwargs": dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
    },
    "hex9res": {
        **cnn_base,
        "env_class": Raw9BinaryEnv,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(),
            kernel_sizes=(),
            strides=(),
            resnet_layers=5,
            resnet_channels=2,
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [128, 128],
    },
    "hex9cnnY": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(16, 128),
            kernel_sizes=(3, 4),
            strides=(2, 1),
            should_normalize=False,
            activation_fn=th.nn.Tanhshrink,
        ),
        "activation_fn": th.nn.ReLU,  # for the fully connected layer, another interesting is th.nn.Tanhshrink
        "net_arch": [128, 128],
    },
    "hex9cnnZ": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(512,),
            kernel_sizes=(6,),
            strides=(3,),
            should_normalize=False,
            activation_fn=th.nn.Tanhshrink,
        ),
        "activation_fn": th.nn.LeakyReLU,  # for the fully connected layer, another interesting is th.nn.Tanhshrink
        "net_arch": [2048, 2048],
    },
    "hex9cnnA": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(128,),
            kernel_sizes=(5,),
            strides=(2,),
            resnet_layers=3,
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [128, 128],
    },
    "hex9cnnB": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(64,),
            kernel_sizes=(5,),
            strides=(2,),
            should_normalize=False,
            activation_fn=th.nn.Tanhshrink,
            resnet_layers=3,
        ),
        "activation_fn": th.nn.LeakyReLU,
        "net_arch": [64, 64],
    },
    "hex9cnnC": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(128,), kernel_sizes=(5,), strides=(2,)
        ),
        "activation_fn": th.nn.LeakyReLU,
        "net_arch": [16, 16],
    },
    "hex9cnnD": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(64, 64),
            kernel_sizes=(3, 2),
            strides=(2, 2),
            activation_fn=th.nn.Tanh,
        ),
        "optimizer_class": th.optim.AdamW,
        "optimizer_kwargs": dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
        "net_arch": [8, 8],
    },
    "hex9cnnE": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(256, 256), kernel_sizes=(5, 3), strides=(2, 1)
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [8, 8],
    },
    "hex9cnnF": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(128, 256), kernel_sizes=(3, 5), strides=(1, 2)
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [8, 8],
    },
    "hex9cnnG": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(64, 32, 32), kernel_sizes=(3, 3, 3)
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [16, 16],
    },
    "hex9cnnR": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_class": HexResnetFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(2,), kernel_sizes=(5,), resnet_layers=1
        ),
        "activation_fn": th.nn.ReLU,
        "net_arch": [128, 128],
    },
    "hex9rawR": {
        **cnn_base,
        "env_class": Raw9BinaryEnv,
        "features_extractor_class": HexResnetRawFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(2,), kernel_sizes=(5,), resnet_layers=1
        ),
        "activation_fn": th.nn.Tanhshrink,
        "net_arch": [162, 162],
    },
    "hex9az": {
        "env_class": Raw9ChannelledEnv,
        "features_extractor_class": AlphaZeroFeaturesExtractor,
        "features_extractor_kwargs": dict(num_actions=1),
    },
}
