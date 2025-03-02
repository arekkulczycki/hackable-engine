# -*- coding: utf-8 -*-
import torch as th

from hackable_engine.training.envs.dummy_env import DummyEnv
from hackable_engine.training.envs.hex.logit_5_graph_env import Logit5GraphEnv
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env
from hackable_engine.training.envs.hex.seq_5_env import Seq5Env
from hackable_engine.training.envs.hex.seq_5_graph_env import Seq5GraphEnv
from hackable_engine.training.envs.hex.seq_7_env import Seq7Env
from hackable_engine.training.envs.hex.seq_7_graph_env import Seq7GraphEnv
from hackable_engine.training.envs.hex.seq_9_graph_env import Seq9GraphEnv
from hackable_engine.training.hyperparams import *
from hackable_engine.training.network.hex_cnn_features_extractor import (
    HexCnnFeaturesExtractor,
)
from hackable_engine.training.network.hex_graph_features_extractor import HexGraphFeaturesExtractor

cnn_base = dict(
    policy="CnnPolicy",
    # optimizer_class=th.optim.AdamW,
    # optimizer_kwargs=dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
    optimizer_class=th.optim.SGD,
    optimizer_kwargs=dict(
        momentum=SGD_MOMENTUM[0], dampening=SGD_DAMPENING[0], nesterov=True
    ),
    share_features_extractor=True,
    features_extractor_class=HexCnnFeaturesExtractor,
    features_extractor_kwargs=dict(
        board_size=7,
        n_filters=(32,),
        kernel_sizes=(3,),
        should_normalize=False,
        activation_fn=th.nn.Tanh,
    ),
    # should_preprocess_obs=False,
    net_arch=[64, 64],
    ortho_init=False,
    log_std_init=th.log(th.tensor(STD_INIT)),
    # use_expln=True,
    # activation_fn=th.nn.Sigmoid,
)

policy_kwargs_map = {
    "default": dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    "dummy": {
        "policy": "MlpPolicy",
        "env_class": DummyEnv,
        "net_arch": [16, 16],
        # "optimizer_class": th.optim.SGD,
        # "optimizer_kwargs": dict(momentum=SGD_MOMENTUM[0], nesterov=True),
        "optimizer_class": th.optim.AdamW,
        "optimizer_kwargs": dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
        "activation_fn": th.nn.ReLU,
        # "ortho_init": True,
        # "log_std_init": th.log(th.tensor(0.66)),
    },
    "hex5raw": {
        "env_class": Seq5Env,
        "board_size": 5,
        "net_arch": [625, 625],
    },
    "hex5graph": {
        "env_class": Seq5GraphEnv,
        "board_size": 5,
        "net_arch": [125, 125],
    },
    "hex5graphlogit": {
        "env_class": Logit5GraphEnv,
        "board_size": 5,
        "net_arch": [125, 125],
    },
    "hex7seq": {
        "env_class": Seq7Env,
        "board_size": 7,
        "net_arch": [343, 343],  #[343, 343],
    },
    "hex7graph": {
        "env_class": Seq7GraphEnv,
        "board_size": 7,
        "net_arch": [72, 72],  # [343, 343],
        "activation_fn": th.nn.ReLU,
        "log_std_init": th.log(th.tensor(0.33)),
    },
    "hex9raw": {
        "policy": "MlpPolicy",
        "env_class": Raw9Env,
        "board_size": 9,
        "net_arch": [729, 729],
        "activation_fn": th.nn.ReLU,
        "log_std_init": th.log(th.tensor(0.33)),
    },
    "hex9graphA": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(18, 18, 18), conv_type="gcn", activation_fn=th.nn.LeakyReLU, use_residuals=True,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81*6, 81*6],
        "log_std_init": th.log(th.tensor(1.0)),
    },
    "hex9graphB": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(18, 18, 18, 18), conv_type="gcn", activation_fn=th.nn.LeakyReLU, use_residuals=True,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(1.0)),
    },
    "hex9graphC": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(12, 12, 12, 12, 12), conv_type="gcn", activation_fn=th.nn.LeakyReLU, use_residuals=True,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(1.0)),
    },
    "hex9graphD": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(6, 6, 6, 6, 6), conv_type="gcn", activation_fn=th.nn.Tanhshrink, use_residuals=True,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(1.0)),
    },
    "hex9graphE": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(36, 36, 36, 36), conv_type="gcn", activation_fn=th.nn.ReLU,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(0.25)),
    },
    "hex9graphF": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(48, 48, 48), conv_type="gcn", activation_fn=th.nn.ReLU,
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
    },
    "hex9graphG": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(6, 3, 1), conv_type="gat",
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(0.5)),
    },
    "hex9graphH": {
        **cnn_base,
        "env_class": Seq9GraphEnv,
        "features_extractor_class": HexGraphFeaturesExtractor,
        "features_extractor_kwargs": dict(
            board_size=9, output_filters=(12, 12, 12), conv_type="res_gated",
        ),
        "activation_fn": th.nn.Tanh,
        "net_arch": [81 * 6, 81 * 6],
        "log_std_init": th.log(th.tensor(0.5)),
    },
    "hex9cnnA": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(128,),
            kernel_sizes=(3,),
            strides=(1,),
            pooling_strides=(2,),
            should_normalize=False,
            activation_fn=th.nn.ReLU,
            # learning_rate=1e-5,
            # reduced_channels=64,
            # resnet_layers=(1,),
        ),
        "log_std_init": th.log(th.tensor(0.42)),
        # "distribution_class": LogNormalDistribution,  # does not exist anymore in sb3
        "activation_fn": th.nn.ReLU,
        "net_arch": [64, 64],
    },
    "hex9cnnB": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(128, 128),
            kernel_sizes=(3, 3),
            strides=(1, 1),
            pooling_strides=(0, 1),
            should_normalize=True,
            activation_fn=th.nn.Tanhshrink,
        ),
        "log_std_init": th.log(th.tensor(0.33)),
        "activation_fn": th.nn.Tanh,
        "net_arch": [64*9, 64*9],
    },
    "hex9cnnC": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(512,),
            kernel_sizes=(5,),
            strides=(1,),
            pooling_strides=(1,),
            should_normalize=False,
            activation_fn=th.nn.Tanhshrink,
        ),
        "log_std_init": th.log(th.tensor(0.42)),
        "activation_fn": th.nn.Tanh,
        "net_arch": [64*9, 64*9],
    },
    "hex9cnnD": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(512, 64),
            kernel_sizes=(5, 3),
            strides=(1, 1),
            should_normalize=False,
            activation_fn=th.nn.Tanhshrink,
        ),
        "log_std_init": th.log(th.tensor(0.42)),
        "activation_fn": th.nn.Tanh,
        "net_arch": [64 * 9, 64 * 9],
    },
    "hex9cnnE": {
        **cnn_base,
        "env_class": Raw9Env,
        "features_extractor_kwargs": dict(
            board_size=9,
            output_filters=(128,),
            kernel_sizes=(5,),
            strides=(1,),
            pooling_strides=(1,),
            should_normalize=False,
            activation_fn=th.nn.ReLU,
            resnet_layers=(7,),
        ),
        "log_std_init": th.log(th.tensor(0.42)),
        "activation_fn": th.nn.ReLU,
        "net_arch": [64*9, 64*9],
    },
}
