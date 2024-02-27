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

from typing import Optional, Tuple

from nptyping import Int, NDArray, Shape, Single
from numpy import (
    asarray,
    float32,
    matmul,
    maximum as np_max,
    minimum as np_min,
    ones,
    sign as np_sign,
)

from hackable_engine.board.chess.chess_board import ChessBoard
from hackable_engine.criteria.evaluation.base_eval import BaseEval

ActionType = NDArray[Shape["10"], Single]

ONES_float32: NDArray[Shape["64"], Single] = ones((64,), dtype=float32)
HALFS_float32: NDArray[Shape["64"], Single] = ones((64,), dtype=float32) / 2
ONES_INT: NDArray[Shape["64"], Single] = ones((64,), dtype=float32)
ONE: float32 = float32(1)
TWO: float32 = float32(2)
FOUR: float32 = float32(4)


class SquareControlEval(BaseEval[ChessBoard]):
    """"""

    DEFAULT_ACTION: ActionType = asarray((
        float32(-0.05),  # king_mobility
        float32(0.05),  # castling_rights
        float32(0.1),  # is_check
        float32(1.00),  # material
        float32(0.01),  # own occupied square control
        float32(0.015),  # opp occupied square control
        float32(0.02),  # empty square control
        # float32(0.015),  # empty square control nominal
        float32(0.02),  # own king proximity square control
        float32(0.025),  # opp king proximity square control
        float32(0.15),  # turn
    ), dtype=float32)
    PARAMS_NUMBER: int = 10

    def get_score(
        self,
        board: ChessBoard,
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
        try:
            king_mobility_int = board.get_king_mobility(True) - board.get_king_mobility(
                False
            )
        except KeyError:
            print(f"king not found in: {board.fen()}")
            print(board.kings)
            print(board.occupied_co)
            # print(board.serialize_position())
            print("continuing")
            return float32(0)
        is_check_value: float32 = (
            -float32(is_check) if board.turn else float32(is_check)
        )  # color is the one who gave the check
        material = (
            float32(board.get_material_no_pawns(True))
            + board.get_material_pawns(True)
            - float32(board.get_material_no_pawns(False))
            - board.get_material_pawns(False)
        )
        # pawn_material: float32 = board.get_material_pawns(True) - board.get_material_pawns(False)

        # how many attacks on each of 64 squares, number of white minus number of black attacks
        square_control_diff: NDArray[Shape["64"], Int]
        square_control_diff_mod_turn: NDArray[Shape["64"], Int]
        (
            square_control_diff,
            square_control_diff_mod_turn,
        ) = board.get_square_control_map_for_both()
        # square_control_diff_sign: NDArray[
        #     Shape["64"], Int
        # ] = np_sign(square_control_diff)
        # square_control_diff_squared: NDArray[
        #     Shape["64"], Int
        # ] = square_control_diff_sign * np_square(square_control_diff)

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
            square_control_diff_mod_turn,
            HALFS_float32,  # defending considered half
        )
        black_material_square_control: NDArray[
            Shape["64"], Single
        ] = black_piece_value_map * np_max(
            square_control_diff_mod_turn,
            -HALFS_float32,  # defending considered half
        )

        white_occupied_square_control: float32 = matmul(
            white_material_square_control, ONES_float32
        )
        black_occupied_square_control: float32 = matmul(
            black_material_square_control, ONES_float32
        )
        empty_square_control: float32 = self._get_empty_square_control(
            # empty_square_control, empty_square_control_nominal = self._get_empty_square_controls(
            board,
            square_control_diff,  # , square_control_diff_sign
        )

        white_king_proximity_square_control: float32
        black_king_proximity_square_control: float32
        (
            white_king_proximity_square_control,
            black_king_proximity_square_control,
        ) = self._get_king_proximity_square_control(board, square_control_diff)

        # a switch is done so that for both players the first param represents the same value
        turn_bonus: float32
        if board.turn:  # white to move, score from blacks perspective
            turn_bonus = ONE
            own_occupied_square_control = black_occupied_square_control
            opp_occupied_square_control = white_occupied_square_control
            own_king_proximity_square_control = black_king_proximity_square_control
            opp_king_proximity_square_control = white_king_proximity_square_control
        else:  # black to move, score from white perspective
            turn_bonus = -ONE
            own_occupied_square_control = white_occupied_square_control
            opp_occupied_square_control = black_occupied_square_control
            own_king_proximity_square_control = white_king_proximity_square_control
            opp_king_proximity_square_control = black_king_proximity_square_control

        params = asarray((
            float32(king_mobility_int),
            castling_rights_value,
            is_check_value,
            material,
            own_occupied_square_control / TWO,
            opp_occupied_square_control / TWO,
            empty_square_control,
            own_king_proximity_square_control / FOUR,
            opp_king_proximity_square_control / FOUR,
            turn_bonus,
        ), dtype=float32)

        return self.calculate_score(action, params)

    @staticmethod
    def _get_empty_square_control(
        board: ChessBoard, square_control_diff: NDArray[Shape["64"], Int]
    ) -> float32:
        empty_square_map: NDArray[Shape["64"], Int] = board.get_empty_square_map()

        return matmul(square_control_diff * empty_square_map, ONES_INT)

    @staticmethod
    def _get_empty_square_controls(
        board: ChessBoard, square_control_diff: NDArray[Shape["64"], Int]
    ) -> Tuple[float32, float32]:
        empty_square_map: NDArray[Shape["64"], Int] = board.get_empty_square_map()

        square_control_diff_nominal: NDArray[Shape["64"], Int] = np_sign(
            square_control_diff
        )

        return matmul(square_control_diff * empty_square_map, ONES_INT), matmul(
            square_control_diff_nominal * empty_square_map, ONES_INT
        )

    @staticmethod
    def _get_king_proximity_square_control(
        board: ChessBoard,
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
