# -*- coding: utf-8 -*-
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType
from nptyping import Int8, NDArray, Shape
from numpy import int8, eye

from arek_chess.training.envs.hex.raw_9_env import Raw9Env


class Raw9x9Env(Raw9Env):
    """"""

    BOARD_SIZE: int = 9
    observation_space = gym.spaces.MultiBinary(BOARD_SIZE ** 2 * 2)

    def step(
        self,
        action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Iterate over all legal moves and evaluate each, storing the move with the best score.

        When generator is exhausted, play the best move, play as opponent and reset the generator.

        To play training moves no engine search is required - a move is selected based on direct evaluation (0 depth).
        """

        if self.best_move is None:
            self.best_move = (self.current_move, action[0])
        else:
            best_score = self.best_move[1]
            current_score = action[0]
            if current_score > best_score:
                self.best_move = (self.current_move, action[0])

        try:
            # if doesn't raise then there are still moves to be evaluated
            self.current_move = next(self.moves)

            # undo last move that was just evaluated
            self.controller.board.pop()

            # push a new move to be evaluated in the next `step`
            self.controller.board.push(self.current_move)

            winner = None

        except StopIteration:
            # all moves evaluated - undo the last and push the best move on the board
            # print("best move: ", self.best_move[0].get_coord(), self.best_move[1])
            self.controller.board.pop()
            self.controller.board.push(self.best_move[0])

            winner = self.controller.board.winner()

            # if the last move didn't conclude the game, now play opponent move and reset generator
            if winner is None:
                # playing against a configured action or an action from self-trained model
                # obs = reshape(self.observation_from_board().astype(float32), (1, 49))
                # opp_action, value = self.opp_model.run(None, {"input": obs})
                # self.controller.make_move(self.prepare_action(opp_action[0]))

                # self.controller.make_move(DEFAULT_ACTION, search_limit=choice((0, 8)))

                self._make_random_move(self.controller.board)
                # self._make_self_trained_move(self.controller.board, self.opp_model, not self.color)

                winner = self.controller.board.winner()

                # if the opponent move didn't conclude, reset generator and play the first option to be evaluated
                if winner is None:
                    self._prepare_child_moves()

        self.obs = self.observation_from_board(self.controller.board)

        self.winner = winner
        reward = self._get_reward(winner)
        self.moves_done += 1

        return self.obs, reward, winner is not None, False, {}

    @staticmethod
    def observation_from_board(board) -> NDArray[Shape["162"], Int8]:
        local: NDArray[Shape["81"], Int8] = board.get_neighbourhood(
            9, should_suppress=True
        ).flatten()
        # fmt: off
        return eye(3, dtype=int8)[local][:, 1:].flatten()  # dummy encoding - 2 columns of 0/1 values, 1 column dropped
        # fmt: on