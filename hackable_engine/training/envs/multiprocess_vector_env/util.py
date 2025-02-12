from collections import OrderedDict
from typing import Any, Union

import numpy as np
from gymnasium import spaces

VecEnvObs = Union[np.ndarray, dict[str, np.ndarray], tuple[np.ndarray, ...]]


def copy_obs_dict(obs):
    assert isinstance(obs, OrderedDict), "unexpected type for observations '{}'".format(
        type(obs)
    )
    return OrderedDict([(k, np.copy(v)) for k, v in obs.items()])


def check_for_nested_spaces(obs_space: spaces.Space) -> None:
    if isinstance(obs_space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = obs_space.spaces.values() if isinstance(obs_space, spaces.Dict) else obs_space.spaces
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )


def dict_to_obs(obs_space: spaces.Space, obs_dict: dict[Any, np.ndarray]) -> VecEnvObs:
    if isinstance(obs_space, spaces.Dict):
        return obs_dict
    elif isinstance(obs_space, spaces.Tuple):
        assert len(obs_dict) == len(obs_space.spaces), "size of observation does not match size of observation space"
        return tuple(obs_dict[i] for i in range(len(obs_space.spaces)))
    else:
        assert set(obs_dict.keys()) == {None}, "multiple observation keys for unstructured observation space"
        return obs_dict[None]
