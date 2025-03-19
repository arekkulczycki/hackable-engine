# -*- coding: utf-8 -*-
import torch as th

from hackable_engine.training.envs.hex.logit_5_graph_env import Logit5GraphEnv
from hackable_engine.training.envs.hex.logit_9_graph_env import Logit9GraphEnv
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env
from hackable_engine.training.envs.hex.seq_5_env import Seq5Env
from hackable_engine.training.envs.hex.seq_5_graph_env import Seq5GraphEnv
from hackable_engine.training.envs.hex.seq_7_env import Seq7Env
from hackable_engine.training.envs.hex.seq_7_graph_env import Seq7GraphEnv
from hackable_engine.training.hyperparams import *

cnn_base = dict(
    policy="CnnPolicy",
    # optimizer_class=th.optim.AdamW,
    # optimizer_kwargs=dict(weight_decay=ADAMW_WEIGHT_DECAY[0]),
    optimizer_class=th.optim.SGD,
    optimizer_kwargs=dict(
        momentum=SGD_MOMENTUM[0], dampening=SGD_DAMPENING[0], nesterov=True
    ),
    share_features_extractor=True,
    # features_extractor_class=HexCnnFeaturesExtractor,
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
    "hex9graphSG": {
        "board_size": 9,
        "env_type": "logit",
        "env_class": Logit9GraphEnv,
        "gnn_arch": [36, 72, 144, 216],
        "mlp_arch": [128],
    },
}
