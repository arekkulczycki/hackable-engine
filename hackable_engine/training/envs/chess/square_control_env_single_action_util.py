# -*- coding: utf-8 -*-
from typing import Tuple

from numpy import (
    float32,
    matmul,
    maximum as np_max,
    minimum as np_min,
    ones,
)
from nptyping import Int, NDArray, Shape, Single

from hackable_engine.board.chess.chess_board import ChessBoard

ONES_float32 = ones((64,), dtype=float32)
HALFS_float32 = ones((64,), dtype=float32) / 2
ONES_INT = ones((64,), dtype=float32)


def _get_king_proximity_square_control(
    board,
    square_control_diff,
) -> Tuple[float32, float32]:
    white_king_proximity_map: NDArray[
        Shape["64"], Single
    ] = board.get_king_proximity_map_normalized(True)
    black_king_proximity_map: NDArray[
        Shape["64"], Single
    ] = board.get_king_proximity_map_normalized(False)

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


def _get_empty_square_control(board: ChessBoard, square_control_diff) -> float32:
    empty_square_map = board.get_empty_square_map()

    return matmul(square_control_diff * empty_square_map, ONES_INT)


def _board_to_obs(
    board: ChessBoard,
) -> Tuple[
    bool,
    bool,
    bool,
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
    float32,
    float32,
]:
    is_check: bool = board.is_check()
    white_castling_right: bool = board.has_castling_rights(True)
    black_castling_right: bool = board.has_castling_rights(False)

    own_king_mobility: float32 = float32(board.get_king_mobility(board.turn) / 8.0)
    opp_king_mobility: float32 = float32(board.get_king_mobility(not board.turn) / 8.0)

    square_control_diff: NDArray[Shape["64"], Int]
    square_control_diff_mod_turn: NDArray[Shape["64"], Single]
    # how many attacks on each of 64 squares, number of white minus number of black attacks
    (
        square_control_diff,
        square_control_diff_mod_turn,
    ) = board.get_square_control_map_for_both()

    own_king_proximity_control: float32
    opp_king_proximity_control: float32
    (
        own_king_proximity_control,
        opp_king_proximity_control,
    ) = _get_king_proximity_square_control(
        board, square_control_diff
    )  # value range from ~ -25 to 25

    material: float32 = float32(board.get_material_no_pawns_both() / 31.0)
    own_pawns: float32 = float32(board.get_pawns_simple_color(board.turn) / 8.0)
    opp_pawns: float32 = float32(board.get_pawns_simple_color(not board.turn) / 8.0)

    # advanced pawns mean more space
    own_space = float32(
        board.get_material_pawns(board.turn) / (8.0 * 3.0)
    )  # 8 times piece value
    opp_space = float32(
        board.get_material_pawns(not board.turn) / (8.0 * 3.0)
    )  # 8 times piece value

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

    white_occupied_square_control: float32 = matmul(white_material_square_control, ONES_float32)
    black_occupied_square_control: float32 = matmul(black_material_square_control, ONES_float32)
    empty_square_control: float32 = _get_empty_square_control(
        # empty_square_control, empty_square_control_nominal = self._get_empty_square_controls(
        board,
        square_control_diff,  # , square_control_diff_sign
    )
    _25 = float32(25)
    _50 = float32(50)
    _60 = float32(60)
    _120 = float32(120)

    return (
        is_check,
        white_castling_right,
        black_castling_right,
        own_king_mobility,
        opp_king_mobility,
        (own_king_proximity_control + _25) / _50,
        (opp_king_proximity_control + _25) / _50,
        material,
        own_pawns,
        opp_pawns,
        own_space,
        opp_space,
        (empty_square_control + _60) / _120,
        (white_occupied_square_control + _60) / _120,
        (black_occupied_square_control + _60) / _120,
    )
