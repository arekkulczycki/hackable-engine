"""
Evaluation by all the attributes obtained from board in an optimized way.

Desired:
[x] is_check
[x] material (with advanced pawn bonus)
[x] occupied square control (with advanced pawn bonus)
[x] empty square control
[x] king proximity square control + discount factors
[ ] turn - bonus equalizer for whose turn to move it is
[ ] threats (x ray)

Observation:
[x] king mobility (king safety)
[x] material on board
[x] pawns on board (how open is the position)
[ ] bishops on board (color)
[ ] space of each player
[ ] own king proximity
[ ] better openness of the position (many pawns locked > ... > many pawns gone)
"""

from typing import Tuple, Optional

from nptyping import NDArray, Shape, Int, Double
from numpy import double, ones, minimum as np_min, maximum as np_max, matmul

from arek_chess.board.board import Board
from arek_chess.criteria.evaluation.base_eval import ActionType, BaseEval

ACTION_TYPE = Tuple[
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
]
ONES_DOUBLE: NDArray[Shape["64"], Double] = ones((64,), dtype=double)
HALFS_DOUBLE: NDArray[Shape["64"], Double] = ones((64,), dtype=double) / 2
ONES_INT: NDArray[Shape["64"], Double] = ones((64,), dtype=double)
TURN_BONUS: double = double(0.15)
TURN_PENALTY: double = double(-0.15)


class SquareControlEval(BaseEval):
    """"""

    DEFAULT_ACTION: ActionType = (
        double(0.15),  # castling_rights
        double(-0.1),  # king_mobility
        double(0.1),  # is_check
        double(1.0),  # material
        double(0.015),  # own occupied square control
        double(0.015),  # opp occupied square control
        double(0.01),  # empty square control
        double(0.01),  # king proximity square control primary
        double(1.0),  # king proximity square control secondary
    )
    ACTION_SIZE: int = 9

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        is_check: bool,
        action: Optional[ActionType] = None,
    ) -> double:
        """
        Get the score evaluation of the given node.

        :param board: board after the move
        """

        if action is None:
            action = self.DEFAULT_ACTION

        castling_rights_int: int = int(board.has_castling_rights(True)) - int(
            board.has_castling_rights(False)
        )
        try:  # TODO: find the bug that's causing it to raise, seems like king is captured in checkmate position
            king_mobility = board.get_king_mobility(True) - board.get_king_mobility(False)
        except KeyError:
            # white king was captured therefore black win...
            return double(-1000) if board.turn else double(1000)
        is_check_int: int = (
            -int(is_check) if board.turn else int(is_check)
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

        white_king_proximity_map: NDArray[
            Shape["64"], Double
        ] = board.get_king_proximity_map_normalized(True)
        black_king_proximity_map: NDArray[
            Shape["64"], Double
        ] = board.get_king_proximity_map_normalized(False)

        white_piece_value_map: NDArray[
            Shape["64"], Double
        ] = board.get_occupied_square_value_map(True)
        black_piece_value_map: NDArray[
            Shape["64"], Double
        ] = board.get_occupied_square_value_map(False)
        empty_square_map: NDArray[Shape["64"], Int] = board.get_empty_square_map()

        # TODO: for attacks on material should be considered 3 options: equal, more white attack or more black attacks
        white_material_square_control: NDArray[
            Shape["64"], Double
        ] = white_piece_value_map * np_min(
            square_control_diff,
            HALFS_DOUBLE,  # defending multiple times considered equal to defending half
        )
        black_material_square_control: NDArray[
            Shape["64"], Double
        ] = black_piece_value_map * np_max(
            square_control_diff,
            -HALFS_DOUBLE,  # defending multiple times considered equal to defending half
        )

        own_occupied_square_control: double = (
            matmul(white_material_square_control, ONES_DOUBLE)
            if board.turn
            else matmul(black_material_square_control, ONES_DOUBLE)
        )
        opp_occupied_square_control: double = (
            matmul(black_material_square_control, ONES_DOUBLE)
            if board.turn
            else matmul(white_material_square_control, ONES_DOUBLE)
        )
        empty_square_control: double = matmul(
            square_control_diff * empty_square_map, ONES_INT
        )
        k: int = 1  # 4
        king_proximity_square_control: double = matmul(
            (
                (
                    white_king_proximity_map ** (1 + k * action[-1])
                    - black_king_proximity_map ** (1 + k * action[-1])
                )
                * square_control_diff
            ),
            ONES_DOUBLE,
        )

        params = (
            double(castling_rights_int),
            double(king_mobility),
            double(is_check_int),
            double(material),
            double(own_occupied_square_control),
            double(opp_occupied_square_control),
            double(empty_square_control),
            double(king_proximity_square_control),
        )

        turn_bonus: double = TURN_BONUS if board.turn else TURN_PENALTY  # TODO: change to param
        return self.calculate_score(action[:-1], params, turn_bonus)

        # n = len(board.move_stack)
        # return self.calculate_score(action[:-1], params, board.turn, n)
