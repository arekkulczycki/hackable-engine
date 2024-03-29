# -*- coding: utf-8 -*-
from collections import deque, defaultdict
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.figure import Figure
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat
from torch.nn import BatchNorm2d, Conv2d, Module

from hackable_engine.training.hyperparams import N_ENVS, N_STEPS


class TensorboardActionHistogramCallback(BaseCallback):
    """
    Custom callback for plotting actions histogram in tensorboard.
    """

    def __init__(self, agent_color: bool, should_log_actions: bool = False):
        """"""

        super().__init__(verbose=0)

        self.agent_color = agent_color
        self.should_log_actions = should_log_actions

        self.openings: defaultdict = defaultdict(lambda: 0)
        self.actions: deque[np.float32] = deque(maxlen=N_ENVS * N_STEPS * 8)
        self.winners: List[int] = []
        self.rewards: List[np.float32] = []

        self.gradients: defaultdict[str, List[np.float32]] = defaultdict(lambda: [])
        """Map of layer name to the list of gradient values since the last rollout."""

    def _on_training_start(self):
        """"""

        self._log_freq = 2 * N_STEPS

        self.tb_formatter = next(
            formatter
            for formatter in self.logger.output_formats
            if isinstance(formatter, TensorBoardOutputFormat)
        )
        self.tb_formatter.writer.default_bins = 100

    def _on_step(self) -> bool:
        """"""

        # pseudo_rewards = []
        for env_info in self.locals["infos"]:
            if self.should_log_actions:
                self.actions.append(env_info["action"])

            winner: Optional[bool] = env_info["winner"]
            if winner is not None:
                int_winner = int(winner) if self.agent_color else int(not winner)

                self.winners.append(int_winner)
                self.rewards.append(env_info["reward"])
                # self.openings[env_info["opening"]] += int_winner if int_winner else -1
            # else:
            # pseudo_rewards.append(env_info["pseudo_reward"])

        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar(
                "rollout/win_rate",
                np.mean(np.asarray(self.winners)),
                self.num_timesteps,
            )
            self.tb_formatter.writer.add_scalar(
                "rollout/reward", np.mean(np.asarray(self.rewards)), self.num_timesteps
            )
            self.winners.clear()
            self.rewards.clear()

            for name, layer in self.model.policy.named_parameters():
                if "network" in name and layer.grad is not None:
                    name_ = f"cnn{name.split('network')[1]}"
                    self.tb_formatter.writer.add_histogram(
                        f"gradients/{name_}",
                        th.nan_to_num(layer.grad, 0.0),
                        max_bins=100,
                    )
                elif "mlp" in name and layer.grad is not None:
                    name_ = f"mlp{name.split('extractor')[1]}"
                    self.tb_formatter.writer.add_histogram(
                        f"gradients/{name_}",
                        th.nan_to_num(layer.grad, 0.0),
                        max_bins=100,
                    )

            self.gradients = defaultdict(lambda: [])

            if self.should_log_actions:
                self.tb_formatter.writer.add_histogram(
                    "train/actions", np.asarray(self.actions), max_bins=100
                )

            # self.tb_formatter.writer.add_figure("time/openings", figure=self._get_openings_figure())

            self.tb_formatter.writer.flush()

        # else:
        #     self.tb_formatter.writer.add_scalar("rollout/pseudo_reward", np.mean(np.asarray(pseudo_rewards)), self.num_timesteps)
        #     self.tb_formatter.writer.flush()

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """

        # print([layer for name, layer in self.model.policy.named_parameters() if "network" in name and "weight" in name])
        # self._collect_gradients()

    def _collect_gradients(self) -> None:
        for name, layer in self.model.policy.named_parameters():
            if "network" in name and layer.grad is not None:
                self.gradients[name.split("network")[1]].append(th.mean(layer.grad).cpu())
            elif "mlp" in name and layer.grad is not None:
                self.gradients[name.split("mlp")[1]].append(th.mean(layer.grad).cpu())

    def _get_openings_figure(self) -> Figure:
        """"""

        fig, ax = plt.subplots()
        p = ax.bar(self.openings.keys(), self.openings.values())
        ax.bar_label(p, label_type="center")
        return fig
