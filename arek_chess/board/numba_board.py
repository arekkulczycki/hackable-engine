# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import typing
import warnings

from chess import (
    Bitboard,
    BB_PAWN_ATTACKS,
    SQUARES,
    _sliding_attacks,
    _carry_rippler,
    _edges,
)
from numba import njit, types, typeof, uint64, int64, NumbaTypeSafetyWarning
from numba.typed import Dict

warnings.simplefilter('ignore', category=NumbaTypeSafetyWarning)


def _attack_table(deltas: typing.Iterable[int]):
    mask_table = []
    attack_table = []

    for square in SQUARES:
        attacks = Dict.empty(
            key_type=types.uint64,
            value_type=types.uint64,
        )

        mask = _sliding_attacks(square, 0, deltas) & ~_edges(square)
        for subset in _carry_rippler(mask):
            v = _sliding_attacks(square, subset, deltas)
            try:
                attacks[subset] = types.uint64(v)
            except OverflowError:
                print("***")
                print(v)
                print(typeof(v))
                raise

        attack_table.append(attacks)
        mask_table.append(mask)

    return mask_table, attack_table


BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-9, -7, 7, 9])
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table([-8, 8])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])

FLAT_BB_PAWN_ATTACKS = [
    item for sublist in BB_PAWN_ATTACKS for item in sublist
]  # BB_PAWN_ATTACKS.flatten()


class BoardNumbaMixin:
    # @njit
    # def attacks_mask(state: Dict, square: types.uint64) -> Bitboard:
    #     bb_square = BB_SQUARES[square]
    #
    #     if bb_square & state["pawns"]:
    #         color = bool(bb_square & state["occupied_w"])
    #         if color:
    #             square += 64
    #         return FLAT_BB_PAWN_ATTACKS[square]
    #     elif bb_square & state["knights"]:
    #         return BB_KNIGHT_ATTACKS[square]
    #     elif bb_square & state["kings"]:
    #         return BB_KING_ATTACKS[square]
    #     else:
    #         attacks = uint64(0)
    #         result = uint64(0)
    #         if bb_square & state["bishops"] or bb_square & state["queens"]:
    #             attacks: types.uint64 = uint64(
    #                 BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & state["occupied"]]
    #             )
    #         if bb_square & state["rooks"] or bb_square & state["queens"]:
    #             ranks: types.uint64 = uint64(
    #                 BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & state["occupied"]]
    #             )
    #             files: types.uint64 = uint64(
    #                 BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & state["occupied"]]
    #             )
    #             ranks_or_files = ranks | files
    #             result = attacks | ranks_or_files
    #             # attacks |= (
    #             #     BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & state["occupied"]]
    #             #     | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & state["occupied"]]
    #             # )
    #         return result

    @staticmethod
    @njit(int64(uint64))
    def get_bit_count(bitboard) -> int:
        n = 0
        bitboard = int64(bitboard)
        while bitboard:
            n += 1
            bitboard &= bitboard - 1
        return n

    @staticmethod
    @njit(int64(uint64))
    def lsb(bitboard: Bitboard) -> int:
        bits = 0
        bitboard = int64(bitboard)
        while bitboard >> bits:
            bits += 1
        return bits
        # return (bb & -bb).bit_length() - 1


    # @njit(int64[:](int64))
    # def scan_forward(bb):
    #     a = []
    #     while bb:
    #         r = bb & -bb
    #         a.append(r.bit_length() - 1)
    #         bb ^= r
    #     return a

    #
    # @njit
    # def get_bit_count_multiplied(attacked_mask, masks):
    #     factors = [1, 3, 3, 4.5, 9]
    #     result = 0
    #     attacked_mask = int64(attacked_mask)
    #     for i in range(5):
    #         result += get_bit_count(attacked_mask & masks[i]) * factors[i]
    #     return result
