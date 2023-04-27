from itertools import cycle
from time import sleep
from typing import Dict, List, Optional

import gym
import numpy
from numpy import float32, ones, matmul

from arek_chess.common.constants import Print
from arek_chess.criteria.evaluation.base_eval import ActionType
from arek_chess.controller import Controller

DEFAULT_ACTION: ActionType = (
    float32(-0.1),  # king_mobility
    float32(0.15),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.015),  # own occupied square control
    float32(0.015),  # opp occupied square control
    float32(0.01),  # empty square control
    float32(0.01),  # own king proximity square control
    float32(1.0),  # opp king proximity square control
    float32(0.15),  # turn
)
MEDIUM_ACTION: ActionType = (
    float32(-0.05),  # king_mobility
    float32(0.075),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.1),  # own occupied square control
    float32(0.1),  # opp occupied square control
    float32(0.0),  # empty square control
    float32(0.0),  # own king proximity square control
    float32(0.0),  # opp king proximity square control
    float32(0.1),  # turn
)
WEAK_ACTION: ActionType = (
    float32(0.0),  # king_mobility
    float32(0.0),  # castling_rights
    float32(0.1),  # is_check
    float32(1.0),  # material
    float32(0.0),  # own occupied square control
    float32(0.0),  # opp occupied square control
    float32(0.0),  # empty square control
    float32(0.0),  # own king proximity square control
    float32(0.0),  # opp king proximity square control
    float32(0.1),  # turn
)
ACTION_SIZE: int = 10
# EQUAL_MIDDLEGAME_FEN = "r3k2r/1ppbqpp1/pb1p1n1p/n3p3/2B1P2B/2PP1N1P/PPQN1PP1/R3K2R w KQkq - 0 12"
# SHARP_MIDDLEGAME_FEN = "rn1qk2r/pp3ppp/2pb4/5b2/3Pp3/4PNB1/PP3PPP/R2QKB1R w KQkq - 0 10"
# ADVANTAGE_MIDDLEGAME_FEN = "r2qk2r/pp1nbppp/2p1pn2/5b2/2BP1B2/2N1PN2/PP3PPP/R2Q1RK1 w kq - 3 9"
# DISADVANTAGE_MIDDLEGAME_FEN = "rn1qkb1r/p3pppp/2p5/1p1n1b2/2pP4/2N1PNB1/PP3PPP/R2QKB1R w KQkq - 0 8"
# fens = cycle([EQUAL_MIDDLEGAME_FEN, SHARP_MIDDLEGAME_FEN, ADVANTAGE_MIDDLEGAME_FEN, DISADVANTAGE_MIDDLEGAME_FEN])


class SquareControlEnv(gym.Env):
    # metadata = {}

    REWARDS: Dict[str, float32] = {
        "*": float32(0.0),
        "1/2-1/2": float32(0.0),
        "1-0": float32(1.0),
        "0-1": float32(-1.0),
    }

    def __init__(self, controller: Optional[Controller] = None):
        super().__init__()

        self.reward_range = (float32(-1.0), float32(1.0))

        self.action_space = self._get_action_space()
        self.observation_space = self._get_observation_space()

        if controller is None:
            self.controller = Controller(
                Print.MOVE,
                search_limit=9,
                memory_action=True,
                in_thread=False,
                timeout=3,
            )
            # self.controller.boot_up(next(fens), self.action_space.sample())
            self.controller.boot_up()
        else:
            self.controller = controller

        self.obs = self.observation()
        self.steps_done = 0

    def _get_action_space(self):
        return gym.spaces.Box(
            numpy.array(
                [-1 if i == 0 else 0 for i in range(ACTION_SIZE)], dtype=float32
            ),
            numpy.array([1 for _ in range(ACTION_SIZE)], dtype=float32),
        )

    def _get_observation_space(self):
        """
        Optimized state of the board.

        [x] own king mobility (king safety)
        [x] opponents king mobility (king safety)
        [x] material on board
        [x] pawns on board (how open is the position)
        [ ] bishops on board (color)
        [ ] space of each player
        [ ] king proximity
        """

        return gym.spaces.Box(
            numpy.array([0 for _ in range(6)], dtype=numpy.float32),
            numpy.array([1 for _ in range(6)], dtype=numpy.float32),
        )

    def step(self, action: ActionType):
        action = self.action_upgrade(action)
        self._run_action(action)

        result = self.controller.board.result()
        if result == "*":
            # playing against a configured action
            self._run_action(MEDIUM_ACTION)

            result = self.controller.board.result()

        self.obs = self.observation()
        reward = self._get_reward(result)
        self.steps_done += 1

        return self.obs, reward, result != "*", {}

    def reset(self):
        self.render()
        # self.controller.restart(fen=next(fens), action=self.action_space.sample())
        self.controller.restart()

        return self.observation()

    def render(self, mode="human", close=False):
        # ...
        print(self.controller.board.fen())
        print(self.steps_done)

    def observation(self):
        return self._board_to_obs()

    def _board_to_obs(self) -> List[float32]:
        board = self.controller.board
        own_king_mobility = float32(board.get_king_mobility(board.turn) / 8.0)
        opp_king_mobility = float32(board.get_king_mobility(not board.turn) / 8.0)
        square_control_diff = board.get_square_control_map_for_both()
        (
            own_king_proximity_control,
            opp_king_proximity_control,
        ) = _get_king_proximity_square_control(board, square_control_diff)
        material = float32(board.get_material_simple_both() / 40.0)
        pawns = float32(board.get_pawns_simple_both() / 16.0)

        return [
            own_king_mobility,
            opp_king_mobility,
            own_king_proximity_control / 8.0,
            opp_king_proximity_control / 8.0,
            material,
            pawns,
        ]

    def _run_action(self, action: ActionType) -> None:
        self.controller.make_move(action)

    def _get_reward(self, result):
        return self.REWARDS[result]

    def _get_intermediate_reward(self):
        return  # maybe some material-based score

    @staticmethod
    def action_upgrade(action: ActionType) -> ActionType:
        # return action

        # artificially make turn value smaller
        # artificially multiply material value for easier random discovery
        return tuple(
            v / 2 if i == 9 else v * 2 if i == 3 else v for i, v in enumerate(action)
        )

    @staticmethod
    def action_downgrade(action: ActionType) -> ActionType:
        return action
        # return tuple(
        #     (v / 2 + 0.5) if i == 1 else v / 2 if i == 3 else v
        #     for i, v in enumerate(action)
        # )


ONES_float32 = ones((64,), dtype=float32)
def _get_king_proximity_square_control(
    board,
    square_control_diff,
):
    """

    :returns: the value at extreme most would be ~8 I suppose...
        (if all squares around king are controlled by 1 player)
    """

    white_king_proximity_map = board.get_king_proximity_map_normalized(True)
    black_king_proximity_map = board.get_king_proximity_map_normalized(False)

    own_occupied_square_control: float32 = (
        matmul(white_king_proximity_map * square_control_diff, ONES_float32)
        if board.turn
        else matmul(black_king_proximity_map * square_control_diff, ONES_float32)
    )
    opp_occupied_square_control: float32 = (
        matmul(white_king_proximity_map * square_control_diff, ONES_float32)
        if not board.turn
        else matmul(black_king_proximity_map * square_control_diff, ONES_float32)
    )
    return own_occupied_square_control, opp_occupied_square_control
