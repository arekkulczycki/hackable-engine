# -*- coding: utf-8 -*-
from typing import List, Tuple

from nptyping import Int, NDArray, Shape
from numpy import (
    float32,
    matmul,
    ones,
)

from hackable_engine.board.chess.chess_board import ChessBoard

ONES_float32 = ones((64,), dtype=float32)
HALFS_float32 = ones((64,), dtype=float32) / 2
ONES_INT = ones((64,), dtype=float32)
_25 = float32(25)
_50 = float32(50)


def _get_king_proximity_square_control(
    board: ChessBoard,
    square_control_diff: NDArray[Shape["64"], Int],
) -> Tuple[float32, float32]:
    """"""

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


def _board_to_obs(board: ChessBoard) -> List[float32]:
    own_king_mobility = float32(board.get_king_mobility(board.turn) / 8.0)
    opp_king_mobility = float32(board.get_king_mobility(not board.turn) / 8.0)

    square_control_diff: NDArray[Shape["64"], Int]
    square_control_diff, _ = board.get_square_control_map_for_both()
    (
        own_king_proximity_control,
        opp_king_proximity_control,
    ) = _get_king_proximity_square_control(
        board, square_control_diff
    )  # value range from ~ -25 to 25

    material = float32(board.get_material_no_pawns_both() / 31.0)
    own_pawns = float32(board.get_pawns_simple_color(board.turn) / 8.0)
    opp_pawns = float32(board.get_pawns_simple_color(not board.turn) / 8.0)

    # advanced pawns mean more space
    own_space = float32(
        board.get_material_pawns(board.turn) / (8.0 * 3.0)
    )  # 8 times piece value
    opp_space = float32(
        board.get_material_pawns(not board.turn) / (8.0 * 3.0)
    )  # 8 times piece value

    return [
        own_king_mobility,
        opp_king_mobility,
        (own_king_proximity_control + _25) / _50,
        (opp_king_proximity_control + _25) / _50,
        material,
        own_pawns,
        opp_pawns,
        own_space,
        opp_space,
    ]
