# -*- coding: utf-8 -*-
import torch as th
from torch import nn
from torch.nn import functional as F

from hackable_engine.common.constants import TH_FLOAT_TYPE
from hackable_engine.training.device import Device
from hackable_engine.training.models import BaseModule


class CNN(BaseModule):

    def __init__(
        self,
        node_count,
        node_features,
        output_size,
        batch_size,
        num_envs,
        cnn_shape,
        cnn_kernels,
        cnn_strides,
        cnn_paddings,
        mlp_shape,
        use_res: bool = False,
        use_norm: bool = False,
        device: Device = Device.XPU,
    ):
        super().__init__()
        self.node_count = node_count
        self.node_features = node_features
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.cnn_shape = cnn_shape
        self.mlp_shape = mlp_shape
        self.use_res = use_res
        self.use_norm = use_norm
        self.device = device

        self.setup_feature_extractor(cnn_shape, cnn_kernels, cnn_strides, cnn_paddings)
        self.setup_mlp(cnn_shape, mlp_shape, output_size)

        self.initialize_cnn_weights()
        self.initialize_mlp_weights()

    def setup_feature_extractor(self, shape, kernels, strides, paddings):
        if self.use_res:
            self.res_proj_0 = nn.Sequential(
                nn.Conv2d(
                    self.node_features, shape[0], kernel_size=1, device=self.device
                ),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.BatchNorm2d(shape[0], dtype=TH_FLOAT_TYPE, device=self.device),
            )
            self.res_proj_1 = nn.Sequential(
                nn.Conv2d(shape[0], shape[1], kernel_size=1, device=self.device),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.BatchNorm2d(shape[1], dtype=TH_FLOAT_TYPE, device=self.device),
            )
            self.res_proj_2 = nn.Sequential(
                nn.Conv2d(shape[1], shape[2], kernel_size=1, device=self.device),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.BatchNorm2d(shape[2], dtype=TH_FLOAT_TYPE, device=self.device),
            )
            self.res_proj_3 = nn.Sequential(
                nn.Conv2d(shape[2], shape[3], kernel_size=1, device=self.device),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.BatchNorm2d(shape[3], dtype=TH_FLOAT_TYPE, device=self.device),
            )
        self.conv1 = nn.Conv2d(
            self.node_features,
            shape[0],
            kernel_size=kernels[0],
            stride=strides[0],
            padding=paddings[0],
            dtype=TH_FLOAT_TYPE,
            device=self.device,
        )
        self.conv2 = nn.Conv2d(
            shape[0],
            shape[1],
            kernel_size=kernels[1],
            stride=strides[1],
            padding=paddings[1],
            dtype=TH_FLOAT_TYPE,
            device=self.device,
        )
        self.conv3 = nn.Conv2d(
            shape[1],
            shape[2],
            kernel_size=kernels[2],
            stride=strides[2],
            padding=paddings[2],
            dtype=TH_FLOAT_TYPE,
            device=self.device,
        )
        # self.conv4 = nn.Conv2d(
        #     shape[2],
        #     shape[3],
        #     kernel_size=kernels[3],
        #     stride=strides[3],
        #     padding=paddings[3],
        #     dtype=TH_FLOAT_TYPE,
        #     device=self.device,
        # )
        if self.use_norm:
            self.norm1 = nn.BatchNorm2d(shape[0], dtype=TH_FLOAT_TYPE, device=self.device)
            self.norm2 = nn.BatchNorm2d(shape[1], dtype=TH_FLOAT_TYPE, device=self.device)
            self.norm3 = nn.BatchNorm2d(shape[2], dtype=TH_FLOAT_TYPE, device=self.device)
            # self.norm4 = nn.BatchNorm2d(shape[3], dtype=TH_FLOAT_TYPE, device=self.device)
        # self.cnn = (self.conv1, self.conv2, self.conv3, self.conv4)
        self.cnn = (self.conv1, self.conv2, self.conv3)

    def setup_mlp(self, cnn_shape, mlp_shape, output_size):
        mlp = []
        prev_size = cnn_shape[-1]
        for size in [*mlp_shape, output_size]:
            mlp.append(nn.Linear(prev_size, size, device=self.device))
            prev_size = size

        self.mlp = tuple(mlp)

    def forward(self, x, *args):
        x = x.flatten(0, 1)
        # x = x.flatten(1, -1)  # will have no effect on 2-dim tensors, will flatten 3-dim into 2-dim
        x = self.extract_features(x)
        x = self.make_decision(x)
        return x.flatten(1, -1)

    def extract_features(self, x):
        # x = x.flatten(0, 1)
        if self.use_res:
            res0 = self.res_proj_0(x)

        x = self.conv1(x)
        if self.use_norm:
            x = self.norm1(x)
        x = F.relu(x)
        if self.use_res:
            x = x + res0
            res1 = self.res_proj_1(x)

        x = self.conv2(x)
        if self.use_norm:
            x = self.norm2(x)
        x = F.relu(x)
        if self.use_res:
            x = x + res1
            res2 = self.res_proj_2(x)

        x = self.conv3(x)
        if self.use_norm:
            x = self.norm3(x)
        x = F.relu(x)
        if self.use_res:
            x = x + res2
            res3 = self.res_proj_3(x)
        #
        # x = self.conv4(x)
        # if self.use_norm:
        #     x = self.norm4(x)
        # x = F.relu(x)
        # if self.use_res:
        #     x = x + res3

        return x.flatten(1, -1)

    def initialize_cnn_weights(self):
        for layer in self.cnn:
            th.nn.init.kaiming_uniform_(
                layer.weight, mode="fan_in", nonlinearity="relu"
            )
            if layer.bias is not None:
                th.nn.init.zeros_(layer.bias)

    def initialize_mlp_weights(self):
        for layer in self.mlp:
            th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            # th.nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="tanh")
            th.nn.init.zeros_(layer.bias)

    # def make_decision(self, x: th.Tensor):
    #     for layer in self.mlp[:-1]:
    #         x = th.tanh(layer(x))
    #     return self.mlp[-1](x)
