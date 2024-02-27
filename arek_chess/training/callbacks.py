# -*- coding: utf-8 -*-
from collections import deque, defaultdict
from typing import List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat

from arek_chess.training.hyperparams import N_ENVS, N_STEPS
import matplotlib.pyplot as plt


class TensorboardActionHistogramCallback(BaseCallback):
    """
    Custom callback for plotting actions histogram in tensorboard.
    """

    def __init__(self, should_log_actions: bool = False):
        """"""

        super().__init__(verbose=0)

        self.should_log_actions = should_log_actions

        self.openings: defaultdict = defaultdict(lambda: 0)
        self.actions: deque[np.float32] = deque(maxlen=N_ENVS)
        self.winners: List[int] = []
        self.rewards: List[np.float32] = []

    def _on_training_start(self):
        """"""

        self._log_freq = N_STEPS

        self.tb_formatter = next(formatter for formatter in self.logger.output_formats if isinstance(formatter, TensorBoardOutputFormat))

    def _on_step(self) -> bool:
        """"""

        # pseudo_rewards = []
        for env_info in self.locals["infos"]:
            if self.should_log_actions:
                self.actions.append(env_info["action"])

            winner: Optional[bool] = env_info["winner"]
            if winner is not None:
                int_winner = int(winner)

                self.winners.append(int_winner)
                self.rewards.append(env_info["reward"])
                # self.openings[env_info["opening"]] += int_winner if int_winner else -1
            # else:
            #     pseudo_rewards.append(env_info["pseudo_reward"])

        if self.n_calls % self._log_freq == 0:
            self.tb_formatter.writer.add_scalar("rollout/win_rate", np.mean(np.asarray(self.winners)), self.num_timesteps)
            self.tb_formatter.writer.add_scalar("rollout/reward", np.mean(np.asarray(self.rewards)), self.num_timesteps)
            self.winners.clear()
            self.rewards.clear()

            if self.should_log_actions:
                self.tb_formatter.writer.add_histogram("train/actions", np.asarray(self.actions), max_bins=25)

            # self.tb_formatter.writer.add_figure("time/openings", figure=self._get_openings_figure())

            self.tb_formatter.writer.flush()

        # else:
        #     self.tb_formatter.writer.add_scalar("rollout/pseudo_reward", np.mean(np.asarray(pseudo_rewards)), self.num_timesteps)
        #     self.tb_formatter.writer.flush()

        return True

    def _get_openings_figure(self):
        """"""

        fig, ax = plt.subplots()
        p = ax.bar(self.openings.keys(), self.openings.values())
        ax.bar_label(p, label_type='center')
        return fig
