from functools import partial
from random import choice

import numpy as np
import onnxruntime as ort
from gymnasium.vector import VectorEnv, SyncVectorEnv

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.multiprocess_vector_env.safe_sync_vector_env import SafeSyncVectorEnv

StepResult = tuple[np.ndarray, float, bool, bool, dict]


class BatchInferenceVectorEnv(SafeSyncVectorEnv):
    def __init__(self, env_fns: list, copy: bool, color: bool, model_variants: list[str]):
        opponent_color = "black" if color else "white"

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_sessions = [
            ort.InferenceSession(
                f"11_{opponent_color}_{variant}.onnx", providers=["CPUExecutionProvider"], sess_options=sess_options
            )
            for variant in model_variants
        ]
        # super().__init__([partial(fn, self.ort_sessions) for fn in env_fns], copy=copy)

        # FIXME: running custom vector env causes numerical instability and eventually nan in observations tensors
        super(SafeSyncVectorEnv, self).__init__(env_fns)
        self.mode = "batch"
        self._initialize_batch_inference(env_fns)

    def _initialize_batch_inference(self, env_fns):
        if self.mode == "batch":
            self.envs = [fn() for fn in env_fns]  # for batch inference
        else:
            self.envs = [fn(self.ort_sessions) for fn in env_fns]  # for single inference
        self.num_envs = len(self.envs)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space
        self.board_size = self.envs[0].BOARD_SIZE
        self.model_switch_interval = self.board_size

        self.ort_session = choice(self.ort_sessions)
        self.autoreset_envs = [False for _ in self.envs]
        self.step_counter = 0

    def reset(self, *args, **kwargs):
        self.ort_session = choice(self.ort_sessions)
        self.autoreset_envs = [False for _ in self.envs]

        obs_list = []
        infos = {}
        for i, env in enumerate(self.envs):
            # first move is random, but it's only once per entire training session
            obs, info = env.reset(*args, **kwargs)
            obs_list.append(obs)

            infos = self._add_info(infos, info, i)

        # return np.stack(obs_list).astype(FLOAT_TYPE), infos
        return np.stack(obs_list), infos

    def step(self, move_positions):
        if self.mode == "batch":
            return self.step_batch(move_positions)
        else:
            return self.step_single(move_positions)

    def step_single(self, move_positions):
        obs_list = []
        rewards = []
        dones = []
        truncs = []
        infos = {}
        for i, (env, move_position, should_reset) in enumerate(zip(self.envs, move_positions, self.autoreset_envs)):
            if should_reset:
                (obs, info), reward, done, trunc = env.reset(), 0.0, False, False
            else:
                obs, reward, done, trunc, info = env.step(move_position)
            self.autoreset_envs[i] = done or trunc
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncs.append(trunc)
            infos = self._add_info(infos, info, i)

        # return (
        #     safe_cast(obs_list),
        #     safe_cast(rewards),
        #     np.array(dones, dtype=np.bool_),
        #     np.array(truncs, dtype=np.bool_),
        #     infos,
        # )

        t = (
            np.array(obs_list, dtype=FLOAT_TYPE),
            np.array(rewards, dtype=FLOAT_TYPE),
            np.array(dones, dtype=np.bool_),
            np.array(truncs, dtype=np.bool_),
            infos,
        )
        has_error = False
        for i, arr in enumerate(t[:-1]):
            if np.isnan(arr).any():
                has_error = True
                print(f"nan in array {i}")
        if has_error:
            raise ValueError("nan in array")
        return t

    def step_batch(self, move_positions):
        self.step_counter += 1
        if self.step_counter % self.model_switch_interval == 0:
            self.ort_session = choice(self.ort_sessions)

        prepare_outputs: list[tuple[int, StepResult | None]] = [
            ((0, None) if should_reset else env.prepare_step(move_c))
            for env, move_c, should_reset in zip(self.envs, move_positions, self.autoreset_envs)
        ]

        inference_indices = []
        inference_inputs = []
        openings: list[int | None] = []
        for i, (env, (n_moves, step_result)) in enumerate(zip(self.envs, prepare_outputs)):
            if n_moves == 0 and step_result is None:  # when autoreset
                opening = choice(env.OPENINGS)
                inference_inputs.append(
                    env.observation_from_board(HexBoard(size=env.BOARD_SIZE, notation=opening, init_move_stack=True))
                )
                inference_indices.append(i)
                openings.append(opening)
            elif step_result is None:  # when legal move was played
                inference_inputs.append(env.observation_from_board(env.board))
                inference_indices.append(i)
                openings.append(None)
            else:  # when illegal move was played
                openings.append(None)
                self.autoreset_envs[i] = True

        inference_outputs = [None for _ in openings]
        inference_outputs_ = (
            self.ort_session.run(None, {"inputs": np.stack(inference_inputs)})[0] if inference_inputs else []
        )
        # put the inference outputs at appropriate places
        for idx, out in zip(inference_indices, inference_outputs_):
            inference_outputs[idx] = out

        obs_list = []
        rewards = []
        dones = []
        truncs = []
        infos = {}
        for i, (env, (n_moves, step_result), model_output, opening) in enumerate(
            zip(self.envs, prepare_outputs, inference_outputs, openings)
        ):
            if opening is not None:
                (obs, info), reward, done, trunc = env.reset(logits=model_output, opening=opening), 0.0, False, False
                self.autoreset_envs[i] = False
            elif step_result:
                obs, reward, done, trunc, info = step_result
            else:
                obs, reward, done, trunc, info = env.finalize_step(n_moves=n_moves, logits=model_output)
                self.autoreset_envs[i] = done or trunc
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            truncs.append(trunc)
            infos = self._add_info(infos, info, i)

        obs_arr = np.array(obs_list, dtype=FLOAT_TYPE)
        if np.isnan(obs_arr).any():
            raise ValueError("nan in obs array")

        return (
            obs_arr,
            # np.array(rewards, dtype=FLOAT_TYPE),
            safe_cast(rewards),
            np.array(dones, dtype=np.bool_),
            np.array(truncs, dtype=np.bool_),
            infos,
        )


def safe_cast(data: list[float | np.ndarray], threshold=1e-38):
    arr = np.array(data, dtype=float)
    return np.where(np.abs(arr) < threshold, 0.0, arr).astype(FLOAT_TYPE)
