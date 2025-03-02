from itertools import zip_longest
from typing import Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
from nptyping import NDArray
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch.nn import AvgPool2d, BatchNorm2d, MaxPool2d

from hackable_engine.training.network.residual_blocks import (
    ResidualBlock,
    ResidualBottleneckBlock,
)


class HexCnnFeaturesExtractor(BaseFeaturesExtractor):
    """
    Module constructed by a number of convolution layers.

    Depending on given arguments each convolution can be amplified by a set of companion layers such as
        - normalization with `BatchNorm2d`
        - ResNet sequence with a number of blocks of choice
        - activation function of choice
        - pooling with `AvgPool2d`, theoretically more appropriate than MaxPool2d for the sake of using all info

    @see https://towardsdatascience.com/conv2d-to-finally-understand-what-happens-in-the-forward-pass-1bbaafb0b148

    output to the next layer is a map of possible placements of the kernel on the input
    https://arxiv.org/pdf/1603.07285v1.pdf
    """

    def __init__(
        self,
        observation_space: gym.Space,
        board_size: int,
        learning_rate: Optional[float] = None,
        output_filters: Tuple[int, ...] = (16,),
        kernel_sizes: Tuple[int, ...] = (3,),
        strides: Tuple[int, ...] = (1,),
        paddings: Tuple[int, ...] = (0,),
        pooling_strides: Tuple[int, ...] = (),
        resnet_layers: int = 0,
        resnet_channels: int = 1,
        reduced_channels: int = 0,
        should_normalize: bool = True,
        activation_fn: Optional[Type[th.nn.Module]] = None,
        should_initialize_weights: bool = True,
    ) -> None:
        device = th.device("xpu")
        obs = th.as_tensor(observation_space.sample()).to(th.float32)

        if output_filters:
            kernels = [*kernel_sizes, *[3 for _ in pooling_strides]]
            features_dim, out_size = self._get_features_number(
                board_size,
                reduced_channels or output_filters[-1],
                kernels,
                strides + pooling_strides,
            )
            # features_dim = 3200  # cnnB
            # features_dim=12800  # for policy D
        else:
            features_dim = obs.size(dim=1) ** 2 * resnet_channels

        super().__init__(observation_space, features_dim)

        n_channels = obs.size(dim=0)
        input_filters = [n_channels, *output_filters[:-1]]

        conv_layers = self._get_convolution_layers(
            input_filters,
            output_filters,
            kernel_sizes,
            strides,
            paddings,
            pooling_strides,
            activation_fn,
            should_normalize,
            resnet_layers,
            resnet_channels,
            device,
        )
        if reduced_channels:
            conv_layers += (
                th.nn.Conv2d(
                    output_filters[-1], reduced_channels, kernel_size=1, bias=False
                ),
                th.nn.Tanhshrink(),
            )

        self.network = th.nn.Sequential(
            *conv_layers,
            th.nn.Flatten(),
        )

        if should_initialize_weights:
            self._initialize_weights()

    @classmethod
    def _get_convolution_layers(
        cls,
        input_filters,
        output_filters,
        kernel_sizes,
        strides,
        paddings,
        pooling_strides,
        activation_fn,
        should_normalize,
        resnet_layers,
        resnet_channels,
        device,
    ):
        layer_generators = [
            (
                th.nn.Conv2d(
                    i_f,
                    o_f,
                    kernel_size=ks,
                    stride=stride,
                    padding=p,
                    bias=False,
                    device=device,
                )
                for i_f, o_f, ks, stride, p in zip(
                    input_filters, output_filters, kernel_sizes, strides, paddings
                )
            )
        ]
        if should_normalize:
            layer_generators.append(
                (BatchNorm2d(o_f, device=device) for o_f in output_filters)
            )
        if activation_fn:
            layer_generators.append((activation_fn() for _ in output_filters))
        if resnet_layers:
            layer_generators.append(
                (
                    cls._get_resnet(
                        ResidualBlock,
                        num_layers,
                        output_filters[-1] if output_filters else resnet_channels,
                        kernel_size=3,
                        device=device,
                    )
                    if num_layers
                    else None
                    for num_layers in resnet_layers
                )
            )
        if pooling_strides:
            layer_generators.append(
                (
                    AvgPool2d(kernel_size=3, stride=s) if s else None
                    for s in pooling_strides
                )
            )

        layers = sum(zip_longest(*layer_generators), ())
        # discard trailing None-layers - effects of zip-longest and possible shorter generators
        return [layer for layer in layers if layer]

    @staticmethod
    def _get_resnet(
        block_cls: Type[Union[ResidualBlock, ResidualBottleneckBlock]],
        num_layers: int,
        num_channels: int,
        kernel_size: int,
        device: th.device,
    ):
        return th.nn.Sequential(
            *(
                block_cls(
                    num_channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                    device=device,
                )
                for _ in range(num_layers)
            )
        )

    def _initialize_weights(self) -> None:
        """
        Initialize weights for layers:
            `Conv2d`: Xavier if using Tanh activation function
            `Linear`: Kaiming which accounts for non-linear activation function like ReLU
        @see https://www.analyticsvidhya.com/blog/2021/05/how-to-initialize-weights-in-neural-networks/
        @see https://towardsdatascience.com/all-ways-to-initialize-your-neural-network-16a585574b52
        @see https://alexkelly.world/posts/initialization_neural_networks/
        """

        modules = list(self.network.modules())
        n_modules = len(modules)

        for i in range(n_modules):
            module = modules[i]

            if isinstance(module, th.nn.Conv2d):
                self._initialize_conv2d_weights_(module, modules, n_modules, i)

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)

            elif isinstance(module, th.nn.Linear):
                th.nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

                if module.bias is not None:
                    th.nn.init.zeros_(module.bias)

            elif isinstance(module, th.nn.BatchNorm2d):
                th.nn.init.constant_(module.weight, 1)
                th.nn.init.constant_(module.bias, 0)

    @staticmethod
    def _initialize_conv2d_weights_(module, modules, n_modules, i):
        """Change in-place weights in the given module, depending on activation function used after the module."""

        if (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.ReLU)) or (
            n_modules > i + 1 and isinstance(modules[i + 1], th.nn.ReLU)
        ):
            th.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="relu"
            )
        elif (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.LeakyReLU)) or (
            n_modules > i + 1 and isinstance(modules[i + 1], th.nn.LeakyReLU)
        ):
            th.nn.init.kaiming_normal_(
                module.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
        elif (n_modules > i + 2 and isinstance(modules[i + 2], th.nn.Tanhshrink)) or (
            n_modules > i + 1 and isinstance(modules[i + 1], th.nn.Tanhshrink)
        ):
            th.nn.init.normal_(module.weight, mean=0, std=0.3)
        else:
            th.nn.init.xavier_normal_(module.weight, gain=0.5)

    @staticmethod
    def _get_features_number(board_size, out_channels, kernel_sizes, strides):
        """Input to the last layer"""

        out_size = board_size
        for kernel_size, stride in zip(kernel_sizes, [s for s in strides if s]):
            out_size = (out_size - kernel_size) // stride + 1

        filters = out_channels * out_size**2
        return filters, out_size

    def forward(self, observations: NDArray) -> th.Tensor:  # th.Tensor) -> th.Tensor:
        """
        :param observations: array of shape (batch_size, board_size, board_size)
        """
        return self.network(observations)
