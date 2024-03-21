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
from hackable_engine.training.envs.hex.raw_9_env import Raw9Env

path = argv[1]
onnx_model_path = argv[2]
device = None  # argv[3]
print("exporting: ", path, " to: ", onnx_model_path)
model = PPO.load(path, device=device or "cpu")


class OnnxablePolicy(th.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.features_extractor = policy.features_extractor
        # self.features_extractor = HexCnnFeaturesExtractor(self.policy.observation_space, 9, (128,), (5,), (2,), th.nn.Sigmoid)
        # print("features dim: ", self.features_extractor.features_dim)
        self.extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        self.value_net = policy.value_net

    def forward(self, observation: th.Tensor):
        """"""
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`

        # observation = preprocess_obs(
        #     observation,
        #     model.policy.observation_space,
        #     normalize_images=model.policy.normalize_images,
        # )
        # print(observation)

        # return self.policy.forward(observation.reshape(1, observation.shape[0], observation.shape[0]))
        # features = HexCnnFeaturesExtractor(self.policy.observation_space(), 9, (128,), (5,), (2,)).forward(observation.reshape(1, observation.shape[0], observation.shape[0]))
        features = self.features_extractor(observation)
        action_hidden, value_hidden = self.extractor(features)
        return self.action_net(action_hidden)  #, self.value_net(value_hidden)
        #return th.nn.functional.relu(self.action_net(action_hidden), inplace=True)


# Example: model = PPO("MlpPolicy", "Pendulum-v1")
onnxable_model = OnnxablePolicy(model.policy)

print("shape: ", model.observation_space.shape)
# obs_sample = model.observation_space.sample()
obs_sample = Raw9Env.observation_space.sample()
print("sample shape: ", obs_sample.shape)
observation = th.from_numpy(obs_sample.astype(np.float32).reshape(1, 1, obs_sample.shape[1], obs_sample.shape[2]))

th.onnx.export(
    onnxable_model,
    observation,
    onnx_model_path,
    opset_version=19,
    input_names=["inputs"],
    dynamic_axes={
        "inputs": {0: "input"},
        "actions": [0]
    },
)

##### Load and test with onnx

onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model)
# from onnxconverter_common import float16
# model_fp16 = float16.convert_float_to_float16(onnx_model)
# model_fp16.save(onnx_model_path)

ort_session_cpu = ort.InferenceSession(
    onnx_model_path, providers=["CPUExecutionProvider"]
)

# numpy_obs = observation.numpy()
# obs_stack = np.stack([numpy_obs for _ in range(20)], axis=0).reshape((20, 1, numpy_obs.shape[2], numpy_obs.shape[3]))
# sb = model.predict(numpy_obs, deterministic=True)
# rt = OrtBackend.run(onnx_model, numpy_obs)
# rts = ort_session_cpu.run(None, {"inputs": obs_stack})
# print(sb, rt, rts)
# exit(0)

ort_session_openvino = ort.InferenceSession(
    onnx_model_path, providers=["OpenVINOExecutionProvider"]
)

from time import perf_counter

numpy_obs = observation.numpy()
print(numpy_obs.shape)

t0 = perf_counter()
for i in range(100):
    action_sb2 = model.predict(numpy_obs, deterministic=True)
print(perf_counter() - t0)
# action_sb2 = model.predict(observation, deterministic=True)

t0 = perf_counter()
inputs = np.stack([numpy_obs for _ in range(20)], axis=0).reshape((20, 1, numpy_obs.shape[2], numpy_obs.shape[3]))
for i in range(100):
    action_onnx = ort_session_cpu.run(None, {"inputs": inputs})
print(perf_counter() - t0)

t0 = perf_counter()
for i in range(100):
    action_onnx_openvino = ort_session_openvino.run(None, {"inputs": inputs})
#    action_onnx = ort_session.run(None, {"input": numpy_obs})
print(perf_counter() - t0)
print(action_sb2)
print(action_onnx)
print(action_onnx_openvino)
