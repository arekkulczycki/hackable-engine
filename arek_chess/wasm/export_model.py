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
import torch as th

from stable_baselines3 import PPO
from torch.nn import functional as F


path = argv[1]
path_out = argv[2]
print("exporting: ", path, " to: ", path_out)
model = PPO.load(path, device="cpu")

class OnnxablePolicy(th.nn.Module):
    def __init__(self, extractor, action_net, value_net):
        super().__init__()
        self.extractor = extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observation):
        """
        !!! should maybe pre-process observation to have it discrete !!!
        """
        # NOTE: You may have to process (normalize) observation in the correct
        #       way before using this. See `common.preprocessing.preprocess_obs`

        from stable_baselines3.common.preprocessing import preprocess_obs
        observation = preprocess_obs(observation, model.policy.observation_space, normalize_images=model.policy.normalize_images)
        # print(observation)

        action_hidden, value_hidden = self.extractor(observation)
        return self.action_net(action_hidden), self.value_net(value_hidden)

# Example: model = PPO("MlpPolicy", "Pendulum-v1")
onnxable_model = OnnxablePolicy(
    model.policy.mlp_extractor, model.policy.action_net, model.policy.value_net
)

observation_size = model.observation_space.shape
# dummy_input = th.randn(1, *observation_size)
dummy_input = th.from_numpy(np.zeros((1, *observation_size)).astype(np.float32))

th.onnx.export(
    onnxable_model,
    dummy_input,
    path_out,
    opset_version=9,
    input_names=["input"],
)

##### Load and test with onnx

import onnx
import onnxruntime as ort
import numpy as np

onnx_model = onnx.load(path_out)
onnx.checker.check_model(onnx_model)

observation = np.zeros((1, *observation_size)).astype(np.float32)
ort_session = ort.InferenceSession(path_out, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])

from time import perf_counter
# t0 = perf_counter()
# for i in range(10000):
#     action_sb2 = model.predict(observation, deterministic=True)
# print(perf_counter() - t0)
action_sb2 = model.predict(observation, deterministic=True)
t0 = perf_counter()
for i in range(10000):
    action_onnx, value = ort_session.run(None, {"input": observation})
print(perf_counter() - t0)
# print(action_sb2)
print(action_onnx)
