# -*- coding: utf-8 -*-
import sys
from typing import Iterator

from arek_chess.board import BitBoard

if sys.version_info[1] <= 9:
    from gmpy2.gmpy2 import popcount  # type: ignore

    def get_bit_count(bb: BitBoard) -> int:
        return popcount(bb)

else:

    def get_bit_count(bb: BitBoard) -> int:
        return bb.bit_count()


def generate_cells(bb: BitBoard) -> Iterator:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


def generate_masks(bb: BitBoard) -> Iterator:
    while bb:
        r = bb & -bb
        yield r
        bb ^= r


# is wrong
# def generate_masks_reversed(bb: BitBoard) -> Iterator:
#     bit = 1 << get_bit_count(bb)
#     while bb <= bit:
#         if bb & bit:
#             yield
#         bit >>= 1


def generate_neighbours_black(bb: BitBoard) -> Iterator:
    while bb:
        r = bb & -bb
        yield r
        bb ^= r


def generate_neighbours_white(bb: BitBoard) -> Iterator:
    while bb:
        r = bb & -bb
        yield r
        bb ^= r
