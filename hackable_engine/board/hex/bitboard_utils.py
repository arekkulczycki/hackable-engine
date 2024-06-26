# -*- coding: utf-8 -*-
import sys
from typing import Iterator, Generator

import numpy as np

from hackable_engine.board import BitBoard

if sys.version_info[1] <= 9:
    from gmpy2.gmpy2 import popcount  # type: ignore

    def get_bit_count(bb: BitBoard) -> int:
        return popcount(bb)

else:

    def get_bit_count(bb: BitBoard) -> int:
        return bb.bit_count()


def generate_cells(bb: BitBoard) -> Iterator:
    """"""

    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


def generate_masks(bb: BitBoard) -> Iterator:
    """"""

    while bb:
        r = bb & -bb
        yield r
        bb ^= r


def get_random_mask(bb: BitBoard, n: int) -> BitBoard:
    """"""

    start_bb = bb

    i = 0
    while bb:
        r = bb & -bb
        if i == n:
            return r
        bb ^= r
        i += 1

    return get_random_mask(start_bb, min(n-i, i))


def int_to_binary_array(n: int, size: int):
    arr = np.zeros(size, dtype=np.int8)
    i = 0
    while n:
        v = n & 1
        if v:
            arr[i] = v
        n >>= 1
        i += 1

    return arr


def int_to_inverse_binary_array(n: int, size: int):
    return np.asarray(list(np.binary_repr(n, width=size)), dtype=np.int8)


def exhaust_generator(g: Generator) -> int:
    """Get the number of elements from a generator."""

    i = 0
    for _ in g:
        i += 1
    return i
