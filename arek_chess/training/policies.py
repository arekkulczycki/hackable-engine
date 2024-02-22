# -*- coding: utf-8 -*-
import torch as th

from arek_chess.training.envs.hex.raw_7_env import Raw7Env
from arek_chess.training.envs.hex.raw_9_env import Raw9Env
from arek_chess.training.envs.hex.raw_9x9_env import Raw9x9Env
from arek_chess.training.hex_cnn_features_extractor import HexCnnFeaturesExtractor

cnn_base = dict(
    policy="CnnPolicy",
   optimizer_class=th.optim.AdamW,
   optimizer_kwargs=dict(weight_decay=1e-2),
#     optimizer_class=th.optim.SGD,
#     optimizer_kwargs=dict(momentum=0.5),
    features_extractor_class=HexCnnFeaturesExtractor,
    features_extractor_kwargs=dict(board_size=7, n_filters=(32,), kernel_sizes=(3,)),
    should_preprocess_obs=False,
    net_arch=[64, 64],
    use_expln=True,
)

# POLICY_KWARGS["activation_fn"] = "tanh"
policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "hex": dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])]),
    "hex7raw": dict(net_arch=dict(pi=[49, 49], vf=[49, 49])),
    "hex7cnnA": {
        **cnn_base,
        "env_class": Raw7Env,
    },
    "hex7cnnB": {
        **cnn_base,
        "env_class": Raw7Env,
        "features_extractor_kwargs": dict(board_size=7, output_filters=(128,), kernel_sizes=(5,)),
        "net_arch": [64, 32],
    },
    "hex7cnnC": {
        **cnn_base,
        "env_class": Raw7Env,
        "features_extractor_kwargs": dict(board_size=7, output_filters=(32, 64), kernel_sizes=(3, 3)),
        "net_arch": [64, 32],
    },
    "hex7cnnD": {
        **cnn_base,
        "env_class": Raw7Env,
        "features_extractor_kwargs": dict(
            board_size=7, output_filters=(32, 64), kernel_sizes=(3, 3), activation_func_class=th.nn.Tanh
        ),
        "net_arch": [64, 32],
    },
    "hex9cnnA": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(128,), kernel_sizes=(5,), strides=(2,)),
    },
    "hex9cnnB": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(64,), kernel_sizes=(3,), strides=(2,)),
    },
    "hex9cnnC": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(128, 128), kernel_sizes=(5, 3), strides=(1, 1)),
    },
    "hex9cnnD": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(128,), kernel_sizes=(5,), strides=(2,), activation_func_class=th.nn.Tanh
        ),
    },
    "hex9cnnE": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(128,), kernel_sizes=(5,), strides=(2,)),
        "net_arch": [128, 128],
    },
    "hex9cnnF": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(128,), kernel_sizes=(5,)),
    },
    "hex9cnnG": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(board_size=9, output_filters=(64, 32, 32), kernel_sizes=(3, 3, 3)),
    },
    "hex9raw": dict(env_class=Raw9x9Env, net_arch=[162, 81]),
    "tight-fit": dict(net_arch=[dict(pi=[10, 16], vf=[16, 10])]),
    "additional-layer": dict(net_arch=[dict(pi=[10, 24, 16], vf=[16, 10])]),
}
