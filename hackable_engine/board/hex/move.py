# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Tuple, Iterable

from hackable_engine.board import BitBoard


@dataclass(slots=True)
class CWCounter:
    black_connectedness: int
    white_connectedness: int
    black_spacing: int
    white_spacing: int


@dataclass(slots=True)
class Move:
    mask: BitBoard
    size: int

    def __str__(self) -> str:
        """"""

        return self.get_coord()

    def __hash__(self) -> int:
        """"""

        return (self.mask << self.size) + (1 << (self.size - 1))

    def uci(self) -> str:
        """"""

        return self.get_coord()

    def get_coord(self) -> str:
        """"""

        bl: int = self.mask.bit_length() - 1
        x = bl % self.size
        y = bl // self.size
        return f"{chr(x + 97)}{y + 1}"

    @classmethod
    def from_coord(cls, coord: str, size: int) -> Move:
        """"""

        return cls(cls.mask_from_coord(coord, size), size)

    @classmethod
    def from_xy(cls, x: int, y: int, size: int) -> Move:
        """"""

        return cls(cls.mask_from_xy(x, y, size), size)

    @staticmethod
    def mask_from_coord(coord: str, size: int) -> BitBoard:
        """"""

        col_str: str
        row_str: str
        g: Tuple[bool, Iterable]

        groups = groupby(coord, str.isalpha)
        col_str, row_str = ("".join(g[1]) for g in groups)

        col = ord(col_str) - 97
        row = int(row_str)

        if col > size or row > size:
            raise ValueError(f"Move coordinate {coord} is outside of given size bounds")

        # a1 => (0, 0) => 0b1
        return 1 << (col + size * (row - 1))

    @staticmethod
    def xy_from_mask(mask: BitBoard, size: int) -> Tuple[int, int]:
        """"""

        bl = mask.bit_length()
        return (bl - 1) % size, (bl - 1) // size

    @staticmethod
    def mask_from_xy(x: int, y: int, size: int) -> BitBoard:
        """"""

        return 1 << (x + size * y)

    @property
    def x(self) -> int:
        """"""

        return self._x(self.mask, self.size)

    @staticmethod
    def _x(mask: BitBoard, size: int):
        """"""

        return (mask.bit_length() - 1) % size

    @property
    def y(self) -> int:
        """"""

        return self._y(self.mask, self.size)

    @staticmethod
    def _y(mask: BitBoard, size: int) -> int:
        """"""

        return (mask.bit_length() - 1) // size
