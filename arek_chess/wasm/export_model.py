# -*- coding: utf-8 -*-
"""
pip install stable_baselines3
pip install gym
pip install shimmy
pip install onnx
pip install onnxruntime
"""

from sys import argv

import numpy as np
import onnx
import torch as th
from stable_baselines3 import PPO

import onnxruntime as ort
from arek_chess.training.envs.hex.raw_7_env import Raw7Env
from arek_chess.training.hex_cnn_features_extractor import HexCnnFeaturesExtractor

path = argv[1]
onnx_model_path = argv[2]
print("exporting: ", path, " to: ", onnx_model_path)
model = PPO.load(path, device="cpu")


class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.features_extractor = policy.features_extractor
        self.extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.value_net = policy.value_net

    def forward(self, observation: th.Tensor):
        """
        !!! should maybe pre-process observation to have it discrete !!!
        """
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`

        # observation = preprocess_obs(
        #     observation,
        #     model.policy.observation_space,
        #     normalize_images=model.policy.normalize_images,
        # )
        # print(observation)

        # return self.policy.forward(observation.reshape(1, observation.shape[0], observation.shape[0]))
        # features = HexCnnFeaturesExtractor(self.policy.observation_space(), 8, 32, 32*5**2, 3).forward(observation.reshape(1, observation.shape[0], observation.shape[0]))
        features = self.features_extractor(observation)
        action_hidden, value_hidden = self.extractor(features)
        # return self.action_net(action_hidden), self.value_net(value_hidden)
        return th.nn.functional.relu(self.action_net(action_hidden), inplace=True)


# Example: model = PPO("MlpPolicy", "Pendulum-v1")
onnxable_model = OnnxablePolicy(model.policy)

print("shape: ", model.observation_space.shape)
# obs_sample = model.observation_space.sample()
obs_sample = Raw7Env.observation_space.sample()
print(obs_sample)
observation = th.from_numpy(obs_sample.astype(np.float32).reshape(1, obs_sample.shape[0], obs_sample.shape[0]))

th.onnx.export(
    onnxable_model,
    observation,
    onnx_model_path,
    opset_version=10,
    input_names=["input"],
)

##### Load and test with onnx

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)

ort_session = ort.InferenceSession(
    onnx_model_path, providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"]
)

from time import perf_counter

numpy_obs = observation.numpy()

t0 = perf_counter()
for i in range(1000):
    action_sb2 = model.predict(numpy_obs, deterministic=True)
print(perf_counter() - t0)
# action_sb2 = model.predict(observation, deterministic=True)
t0 = perf_counter()
for i in range(1000):
    action_onnx = ort_session.run(None, {"input": numpy_obs})
print(perf_counter() - t0)
print(action_sb2)
print(action_onnx)
