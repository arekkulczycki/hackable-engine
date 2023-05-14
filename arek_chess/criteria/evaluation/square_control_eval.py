"""
Evaluation by all the attributes obtained from board in an optimized way.

Desired:
[x] is_check
[x] material (with advanced pawn bonus)
[x] occupied square control (with advanced pawn bonus)
[x] empty square control
[x] king proximity square control + discount factors
[x] turn - bonus equalizer for whose turn to move it is
[ ] threats (x ray)

Observation:
[x] king mobility (king safety?)
[x] material on board
[x] pawns on board (how open is the position)
[x] king proximity square control
[ ] bishops on board (color)
[ ] space of each player
[ ] better openness of the position (many pawns locked > ... > many pawns gone)
"""

from typing import Tuple, Optional

from nptyping import Int, NDArray, Shape, Single
from numpy import float32, ones, minimum as np_min, maximum as np_max, matmul

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.base_eval import ActionType, BaseEval

ACTION_TYPE = Tuple[
    float32,
    float32,
    float32,
    float32,
    float32,
    float32,
    float32,
    float32,
    float32,
    float32,
]
ONES_float32: NDArray[Shape["64"], Single] = ones((64,), dtype=float32)
HALFS_float32: NDArray[Shape["64"], Single] = ones((64,), dtype=float32) / 2
ONES_INT: NDArray[Shape["64"], Single] = ones((64,), dtype=float32)
ONE: float32 = float32(1)


class SquareControlEval(BaseEval):
    """"""

    DEFAULT_ACTION: ActionType = (
        float32(-0.05),  # king_mobility
        float32(0.05),  # castling_rights
        float32(0.1),  # is_check
        float32(1.0),  # material
        float32(0.0075),  # own occupied square control
        float32(0.0125),  # opp occupied square control
        float32(0.01),  # empty square control
        float32(0.01),  # own king proximity square control
        float32(0.02),  # opp king proximity square control
        float32(0.15),  # turn
    )
    ACTION_SIZE: int = 10

    def get_score(
        self,
        board: Board,
        is_check: bool,
        action: Optional[ActionType] = None,
    ) -> float32:
        """
        Get the score evaluation of the given node.

        :param board: board after the move
        """

        if action is None:
            action = self.DEFAULT_ACTION

        castling_rights_value: float32 = float32(
            board.has_castling_rights(True)
        ) - float32(board.has_castling_rights(False))
        king_mobility_int = board.get_king_mobility(True) - board.get_king_mobility(
            False
        )
        is_check_value: float32 = (
            -float32(is_check) if board.turn else float32(is_check)
        )  # color is the one who gave the check
        material = (
            board.get_material_no_pawns(True)
            + board.get_material_pawns(True)
            - board.get_material_no_pawns(False)
            - board.get_material_pawns(False)
        )

        # how many attacks on each of 64 squares, number of white minus number of black attacks
        square_control_diff: NDArray[
            Shape["64"], Int
        ] = board.get_square_control_map_for_both()

        white_piece_value_map: NDArray[Shape["64"], Single]
        black_piece_value_map: NDArray[Shape["64"], Single]
        (
            white_piece_value_map,
            black_piece_value_map,
        ) = board.get_occupied_square_value_map_for_both()

        # TODO: for attacks on material should be considered 3 options: equal, more white attack or more black attacks
        white_material_square_control: NDArray[
            Shape["64"], Single
        ] = white_piece_value_map * np_min(
            square_control_diff,
            HALFS_float32,  # defending multiple times considered equal to defending half
        )
        black_material_square_control: NDArray[
            Shape["64"], Single
        ] = black_piece_value_map * np_max(
            square_control_diff,
            -HALFS_float32,  # defending multiple times considered equal to defending half
        )

        white_occupied_square_control = matmul(
            white_material_square_control, ONES_float32
        )
        black_occupied_square_control = matmul(
            black_material_square_control, ONES_float32
        )
        empty_square_control: float32 = self._get_empty_square_control(
            board, square_control_diff
        )

        white_king_proximity_square_control: float32
        black_king_proximity_square_control: float32
        (
            white_king_proximity_square_control,
            black_king_proximity_square_control,
        ) = self._get_king_proximity_square_control(board, square_control_diff)

        # a switch is done so that for both players the first param represents the same value
        turn_bonus: float32
        if board.turn:  # turn after the move
            turn_bonus = ONE
            own_occupied_square_control = black_occupied_square_control
            opp_occupied_square_control = white_occupied_square_control
            own_king_proximity_square_control = black_king_proximity_square_control
            opp_king_proximity_square_control = white_king_proximity_square_control
        else:
            turn_bonus = -ONE
            own_occupied_square_control = white_occupied_square_control
            opp_occupied_square_control = black_occupied_square_control
            own_king_proximity_square_control = white_king_proximity_square_control
            opp_king_proximity_square_control = black_king_proximity_square_control

        params = (
            float32(king_mobility_int),
            castling_rights_value,
            is_check_value,
            float32(material),
            own_occupied_square_control,
            opp_occupied_square_control,
            empty_square_control,
            own_king_proximity_square_control,
            opp_king_proximity_square_control,
            turn_bonus,
        )
        return self.calculate_score(action, params)

        # n = len(board.move_stack)
        # return self.calculate_score(action[:-1], params, board.turn, n)

    @staticmethod
    def _get_empty_square_control(
        board: Board, square_control_diff: NDArray[Shape["64"], Int]
    ) -> float32:
        empty_square_map: NDArray[Shape["64"], Int] = board.get_empty_square_map()
        return matmul(square_control_diff * empty_square_map, ONES_INT)

    @staticmethod
    def _get_king_proximity_square_control(
        board: Board,
        square_control_diff: NDArray[Shape["64"], Int],
    ) -> Tuple[float32, float32]:
        """
        :returns: the value at most is around
            (if all squares around king are controlled by 1 player)
        """

        white_king_proximity_map: NDArray[
            Shape["64"], Single
        ] = board.get_king_proximity_map_normalized(True)
        black_king_proximity_map: NDArray[
            Shape["64"], Single
        ] = board.get_king_proximity_map_normalized(False)

        white_king_proximity_square_control = matmul(
            white_king_proximity_map * square_control_diff, ONES_float32
        )
        black_king_proximity_square_control = matmul(
            black_king_proximity_map * square_control_diff, ONES_float32
        )
        return white_king_proximity_square_control, black_king_proximity_square_control

    # own_occupied_square_control: float32 = (
    #     matmul(white_king_proximity_map * square_control_diff, ONES_float32)
    #     if board.turn  # board after the move
    #     else matmul(black_king_proximity_map * square_control_diff, ONES_float32)
    # )
    # opp_occupied_square_control: float32 = (
    #     matmul(white_king_proximity_map * square_control_diff, ONES_float32)
    #     if not board.turn
    #     else matmul(black_king_proximity_map * square_control_diff, ONES_float32)
    # )
    # return own_occupied_square_control, opp_occupied_square_control


# def _get_king_proximity_square_control(
#     board: Board,
#     square_control_diff: NDArray[Shape["64"], Int],
#     action: ActionType,
# ) -> float32:
#     """"""
#
#     white_king_proximity_map: NDArray[
#         Shape["64"], Single
#     ] = board.get_king_proximity_map_normalized(True)
#     black_king_proximity_map: NDArray[
#         Shape["64"], Single
#     ] = board.get_king_proximity_map_normalized(False)
#
#     k: int = 1  # 4
#     exponent: float32 = action[-1]
#     return matmul(
#         (
#             (
#                 white_king_proximity_map ** (1 + k * exponent)
#                 - black_king_proximity_map ** (1 + k * exponent)
#             )
#             * square_control_diff
#         ),
#         ONES_float32,
#     )
