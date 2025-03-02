# -*- coding: utf-8 -*-
from typing import SupportsFloat, Any

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from gymnasium.envs.registration import register

from hackable_engine.board.hex.serializers import BoardShapeError
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.base_env import BaseEnv

ZERO: FLOAT_TYPE = FLOAT_TYPE(0)
ONE: FLOAT_TYPE = FLOAT_TYPE(1)
MINUS_ONE: FLOAT_TYPE = FLOAT_TYPE(-1)


class Seq7Env(BaseEnv):
    """"""

    ENV_NAME = "seq7hex"

    observation_space = gym.spaces.Box(
        -1, 1, shape=(1, 7, 7), dtype=FLOAT_TYPE
    )  # should be int8
    action_space = gym.spaces.Box(MINUS_ONE, ONE, shape=(1,), dtype=FLOAT_TYPE)
    """
    Actions: eval, is_bridge_vert, is_bridge_diag  
    """

    def __init__(self, *args, **kwargs):
        """"""

        super().__init__(*args, **kwargs)
        self.BOARD_SIZE: int = 7
        self.MAX_MOVES: int = self.BOARD_SIZE**2
        self.AUXILIARY_REWARD_PER_MOVE = 0.5 * self.REWARDS[True] / (self.MAX_MOVES / 2)
        """Getting auxiliary reward on every move totals to a % of a win."""
        self.AUXILIARY_REWARD_PER_STEP = self.AUXILIARY_REWARD_PER_MOVE / self.MAX_MOVES
        self.DECISIVE_DISTANCE_ADVANTAGE: int = 3
        # fmt: off
        self.OPENINGS = [
            "a1", "a2", "a3", "a4", "a5", "a6", "a7",
            "g1", "g2", "g3", "g4", "g5", "g6", "g7",
            "d2", "d6",
        ]
        # fmt: on
        self.guessed_bridges = 0

    def _get_intermediate_reward(self, n_moves):
        if self.did_force_stop:
            distance = self._get_distance_score(n_moves, early_finish=False)
            if (distance > 0 and self.color) or (distance < 0 and not self.color):
                return FLOAT_TYPE(self.AUXILIARY_REWARD_PER_MOVE)
            else:
                return FLOAT_TYPE(-self.AUXILIARY_REWARD_PER_MOVE)
        return ZERO
        # if n_moves <= 24:
        #     score = self._get_distance_score_perf(n_moves, early_finish=False)
        # else:
        #     score = self._get_distance_score(n_moves, early_finish=False)
        # relative_score = score - self.last_intermediate_score
        # self.last_intermediate_score = score
        #
        # if not self.color:
        #     relative_score *= MINUS_ONE
        #
        # return relative_score

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # win_percentage = (
        #     np.mean(self.results) if len(self.results) > N_ENVS / 2 else 1  # 0.4
        # )
        # if win_percentage < 0.75:
        #     self.guess_bridges(action)

        return super().step(action)

    def guess_bridges(self, action):
        is_bridge_vert_score = action[1]
        try:
            is_bridge_vert = self.is_last_move_bridge_vert()
        except BoardShapeError:
            is_bridge_vert = False
        is_bridge_diag_score = action[2]
        try:
            is_bridge_diag = self.is_last_move_bridge_diag()
        except BoardShapeError:
            is_bridge_diag = False

        correct_positive_vert = is_bridge_vert and is_bridge_vert_score > 0.1
        correct_positive_diag = is_bridge_diag and is_bridge_diag_score > 0.1
        false_positive_vert = not is_bridge_vert and is_bridge_vert_score > 0.1
        false_positive_diag = not is_bridge_diag and is_bridge_diag_score > 0.1

        correct_negative_vert = not is_bridge_vert and is_bridge_vert_score < -0.1
        correct_negative_diag = not is_bridge_diag and is_bridge_diag_score < -0.1
        false_negative_vert = is_bridge_vert and is_bridge_vert_score < -0.1
        false_negative_diag = is_bridge_diag and is_bridge_diag_score < -0.1

        if false_positive_vert or false_positive_diag:
            self.guessed_bridges -= 1
            self.auxiliary_reward -= 5 * self.AUXILIARY_REWARD_PER_STEP

        elif false_negative_diag or false_negative_vert:
            self.guessed_bridges -= 1
            self.auxiliary_reward -= 5 * self.AUXILIARY_REWARD_PER_STEP

        elif correct_positive_vert or correct_positive_diag:
            self.guessed_bridges += 1
            self.auxiliary_reward += 5 * self.AUXILIARY_REWARD_PER_STEP

        elif correct_negative_vert or correct_negative_diag:
            self.guessed_bridges += 1
            self.auxiliary_reward += 0.5 * self.AUXILIARY_REWARD_PER_STEP

    def is_last_move_bridge_diag(self):
        """For the moment considers own moves only, therefore assumes `color=self.color`"""
        if not self.controller.board.move_stack:
            return False

        last_mask = self.controller.board.move_stack[-1].mask
        oc_co = self.controller.board.occupied_co[self.color]
        unoc = self.controller.board.unoccupied

        if oc_co & self.controller.board.bridge_diag_left(last_mask):
            if (
                unoc
                & self.controller.board.cell_left(last_mask)
                & self.controller.board.cell_up(last_mask)
            ):
                return True

        if oc_co & self.controller.board.bridge_diag_right(last_mask):
            if (
                unoc
                & self.controller.board.cell_right(last_mask)
                & self.controller.board.cell_down(last_mask)
            ):
                return True

        return False

    def is_last_move_bridge_vert(self):
        """For the moment considers own moves only, therefore assumes `color=self.color`"""
        if not self.controller.board.move_stack:
            return False

        last_mask = self.controller.board.move_stack[-1].mask
        oc_co = self.controller.board.occupied_co[self.color]
        unoc = self.controller.board.unoccupied

        if self.color:
            if oc_co & self.controller.board.bridge_white_left(last_mask):
                if (
                    unoc
                    & self.controller.board.cell_left(last_mask)
                    & self.controller.board.cell_downleft(last_mask)
                ):
                    return True

            if oc_co & self.controller.board.bridge_white_right(last_mask):
                if (
                    unoc
                    & self.controller.board.cell_right(last_mask)
                    & self.controller.board.cell_upright(last_mask)
                ):
                    return True
        else:
            if oc_co & self.controller.board.bridge_black_down(last_mask):
                if (
                    unoc
                    & self.controller.board.cell_down(last_mask)
                    & self.controller.board.cell_downleft(last_mask)
                ):
                    return True

            if oc_co & self.controller.board.bridge_black_up(last_mask):
                if (
                    unoc
                    & self.controller.board.cell_up(last_mask)
                    & self.controller.board.cell_upright(last_mask)
                ):
                    return True

        return False


    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        # print("guessed bridges", self.guessed_bridges)
        self.guessed_bridges = 0
        return super().reset()


register(
    id="Seq7Env",
    entry_point="hackable_engine.training.envs.hex.seq_7_env:Seq7Env",
)
