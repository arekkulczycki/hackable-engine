# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import reduce
from itertools import groupby
from operator import ior
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from arek_chess.board import BitBoard, GameBoardBase
from arek_chess.board.hex.mixins import BoardShapeError
from arek_chess.board.hex.mixins.bitboard_utils import generate_masks
from arek_chess.board.hex.mixins.hex_board_serializer_mixin import (
    HexBoardSerializerMixin,
)
from numpy import mean

Cell = int

SIZE = 13

# fmt: off
CELLS_13 = [
    A1,B1,C1,D1,E1,F1,G1,H1,I1,J1,K1,L1,M1,
    A2,B2,C2,D2,E2,F2,G2,H2,I2,J2,K2,L2,M2,
    A3,B3,C3,D3,E3,F3,G3,H3,I3,J3,K3,L3,M3,
    A4,B4,C4,D4,E4,F4,G4,H4,I4,J4,K4,L4,M4,
    A5,B5,C5,D5,E5,F5,G5,H5,I5,J5,K5,L5,M5,
    A6,B6,C6,D6,E6,F6,G6,H6,I6,J6,K6,L6,M6,
    A7,B7,C7,D7,E7,F7,G7,H7,I7,J7,K7,L7,M7,
    A8,B8,C8,D8,E8,F8,G8,H8,I8,J8,K8,L8,M8,
    A9,B9,C9,D9,E9,F9,G9,H9,I9,J9,K9,L9,M9,
    A10,B10,C10,D10,E10,F10,G10,H10,I10,J10,K10,L10,M10,
    A11,B11,C11,D11,E11,F11,G11,H11,I11,J11,K11,L11,M11,
    A12,B12,C12,D12,E12,F12,G12,H12,I12,J12,K12,L12,M12,
    A13,B13,C13,D13,E13,F13,G13,H13,I13,J13,K13,L13,M13,
] = range(SIZE*SIZE)
BB_CELLS_13 = [
    BB_A1,BB_B1,BB_C1,BB_D1,BB_E1,BB_F1,BB_G1,BB_H1,BB_I1,BB_J1,BB_K1,BB_L1,BB_M1,
    BB_A2,BB_B2,BB_C2,BB_D2,BB_E2,BB_F2,BB_G2,BB_H2,BB_I2,BB_J2,BB_K2,BB_L2,BB_M2,
    BB_A3,BB_B3,BB_C3,BB_D3,BB_E3,BB_F3,BB_G3,BB_H3,BB_I3,BB_J3,BB_K3,BB_L3,BB_M3,
    BB_A4,BB_B4,BB_C4,BB_D4,BB_E4,BB_F4,BB_G4,BB_H4,BB_I4,BB_J4,BB_K4,BB_L4,BB_M4,
    BB_A5,BB_B5,BB_C5,BB_D5,BB_E5,BB_F5,BB_G5,BB_H5,BB_I5,BB_J5,BB_K5,BB_L5,BB_M5,
    BB_A6,BB_B6,BB_C6,BB_D6,BB_E6,BB_F6,BB_G6,BB_H6,BB_I6,BB_J6,BB_K6,BB_L6,BB_M6,
    BB_A7,BB_B7,BB_C7,BB_D7,BB_E7,BB_F7,BB_G7,BB_H7,BB_I7,BB_J7,BB_K7,BB_L7,BB_M7,
    BB_A8,BB_B8,BB_C8,BB_D8,BB_E8,BB_F8,BB_G8,BB_H8,BB_I8,BB_J8,BB_K8,BB_L8,BB_M8,
    BB_A9,BB_B9,BB_C9,BB_D9,BB_E9,BB_F9,BB_G9,BB_H9,BB_I9,BB_J9,BB_K9,BB_L9,BB_M9,
    BB_A10,BB_B10,BB_C10,BB_D10,BB_E10,BB_F10,BB_G10,BB_H10,BB_I10,BB_J10,BB_K10,BB_L10,BB_M10,
    BB_A11,BB_B11,BB_C11,BB_D11,BB_E11,BB_F11,BB_G11,BB_H11,BB_I11,BB_J11,BB_K11,BB_L11,BB_M11,
    BB_A12,BB_B12,BB_C12,BB_D12,BB_E12,BB_F12,BB_G12,BB_H12,BB_I12,BB_J12,BB_K12,BB_L12,BB_M12,
    BB_A13,BB_B13,BB_C13,BB_D13,BB_E13,BB_F13,BB_G13,BB_H13,BB_I13,BB_J13,BB_K13,BB_L13,BB_M13,
] = [1 << sq for sq in CELLS_13]
# fmt: on

VEC_1: BitBoard = 2**13 - 1


@dataclass
class CWCounter:
    black_connectedness: int
    white_connectedness: int
    black_wingspan: int
    white_wingspan: int


@dataclass
class Move:
    mask: BitBoard

    def get_coord(self) -> str:
        """"""

        raise NotImplementedError

    @classmethod
    def from_cord(cls, coord: str, size: int) -> Move:
        """"""

        col_str: str
        row_str: str
        g: Tuple[bool, Iterable]

        groups = groupby(coord, str.isalpha)
        row_str, col_str = ("".join(g[1]) for g in groups)

        # a1 => (0, 0) => 0b1
        return cls(1 << ((int(col_str) - 1) + (ord(row_str) - 97) * size))


class HexBoard(GameBoardBase, HexBoardSerializerMixin):
    """
    Handling the hex board and calculating features of a position.
    """

    turn: bool
    """The side to move True for white, False for black."""

    move_stack: List[Move]
    """List of moves on board from first to last."""

    occupied_co: Dict[bool, BitBoard]
    unoccupied: BitBoard

    def __init__(self, size=13) -> None:
        """"""

        super().__init__()

        self.size = size
        self.bb_rows: List[BitBoard] = [
            reduce(ior, [1 << (col + row * size) for col in range(size)])
            for row in range(size)
        ]
        self.bb_cols: List[BitBoard] = [
            reduce(ior, [(1 << col) << (row * size) for row in range(size)])
            for col in range(size)
        ]
        self.vertical_coeff = 2**self.size
        self.diagonal_coeff = 2 ** (self.size - 1)
        self.reset_board()

    def reset_board(self) -> None:
        """"""

        self.turn = False
        self.occupied_co = {False: 0, True: 0}
        self.unoccupied = 0

    def color_at(self, cell: Cell) -> Optional[bool]:
        """"""

        mask = BB_CELLS_13[cell]
        return self.color_at_mask(mask)

    def color_at_mask(self, mask: BitBoard) -> Optional[bool]:
        """"""

        if self.occupied_co[False] & mask:
            return False
        elif self.occupied_co[True] & mask:
            return True
        else:
            return None

    @property
    def occupied(self) -> BitBoard:
        """"""

        return self.occupied_co[False] | self.occupied_co[True]

    def is_adjacent(self, cell_1: Cell, cell_2: Cell):
        """"""

        if cell_1 is cell_2:
            return False

        return self.is_adjacent_mask(BB_CELLS_13[cell_1], BB_CELLS_13[cell_2])

    def is_adjacent_mask(self, mask_1: BitBoard, mask_2: BitBoard) -> bool:
        """"""

        larger, smaller = sorted((mask_1, mask_2))
        ratio = larger // smaller

        return ratio in {2, self.diagonal_coeff, self.vertical_coeff}

    def cell_right(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first column on the left
        if mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift right {bin(mask)}")

        # cells are ordered opposite direction to bits in bitboard
        return mask << 1

    def cell_left(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first column on the left
        if mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift left {bin(mask)}")

        # cells are ordered opposite direction to bits in bitboard
        return mask >> 1

    def cell_down(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first row on top
        if mask & self.bb_rows[self.size - 1]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask << self.size

    def cell_up(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask >> self.size

    def cell_downleft(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (self.size - 1)

    def cell_upright(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask >> (self.size - 1)

    def bridge_diag_right(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (self.size + 1)

    def bridge_diag_left(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0] or mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (self.size + 1)

    def bridge_black_down(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (2 * self.size - 1)

    def bridge_black_up(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (2 * self.size - 1)

    def bridge_white_right(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (self.size - 2)

    def bridge_white_left(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (self.size - 2)

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        """"""

        return self.winner() is not None

    def winner(self) -> Optional[bool]:
        """"""

        if self.turn and self.is_black_win():  # last move was black
            return False

        elif not self.turn and self.is_white_win():  # last move was white
            return True

        return None

    def is_black_win(self) -> bool:
        """"""

        blacks: BitBoard = self.occupied_co[False]

        # check there is a black stone on every row
        if not all((blacks & row for row in self.bb_rows)):
            return False

        # find connection from top to bottom
        for mask in generate_masks(blacks & self.bb_rows[0]):
            if self.is_connected_to_bottom(mask):
                return True

        return False

    def is_connected_to_bottom(self, mask: BitBoard, visited: BitBoard = 0) -> bool:
        """
        Recurrent way of finding if a stone is connected to bottom.
        """

        if mask & self.bb_rows[self.size - 1]:
            return True

        visited |= mask

        for neighbour in self.generate_neighbours_black(mask, visited):
            if self.is_connected_to_bottom(neighbour, visited):
                return True

        return False

    def generate_neighbours_black(self, mask: BitBoard, visited: BitBoard) -> Iterator:
        """
        Generate in optimized order to find black connection.
        """

        blacks: BitBoard = self.occupied_co[False]

        for f in (
            self.cell_right,
            self.cell_down,
            self.cell_downleft,
            self.cell_left,
            self.cell_upright,
            self.cell_up,
        ):
            try:
                neighbour_cell = f(mask)
            except BoardShapeError:
                continue
            else:
                if neighbour_cell & blacks & ~visited:
                    yield neighbour_cell

    def is_white_win(self) -> bool:
        """"""

        whites: BitBoard = self.occupied_co[True]

        # check there is a white stone on every column
        if not all((whites & col for col in self.bb_cols)):
            return False

        # find connection from left to right
        for mask in generate_masks(whites & self.bb_cols[0]):
            if self.is_connected_to_right(mask):
                return True

        return False

    def is_connected_to_right(self, mask: BitBoard, visited: BitBoard = 0) -> bool:
        """
        Recurrent way of finding if a stone is connected to bottom.
        """

        if mask & self.bb_cols[self.size - 1]:
            return True

        visited |= mask

        for neighbour in self.generate_neighbours_white(mask, visited):
            if self.is_connected_to_right(neighbour, visited):
                return True

        return False

    def generate_neighbours_white(
        self, mask: BitBoard, visited: BitBoard
    ) -> Generator[BitBoard, None, None]:
        """
        Generate in optimized order to find white connection.
        """

        whites: BitBoard = self.occupied_co[True]

        for f in (
            self.cell_down,
            self.cell_right,
            self.cell_upright,
            self.cell_up,
            self.cell_downleft,
            self.cell_left,
        ):
            try:
                neighbour_cell = f(mask)
            except BoardShapeError:
                continue
            else:
                if neighbour_cell & whites & ~visited:
                    yield neighbour_cell

    def generate_adjacent_cells(
        self,
        mask: BitBoard,
        visited: BitBoard = 0,
        empty_only: bool = False,
        direction: Optional[bool] = None,
    ) -> Generator[BitBoard, None, None]:
        """
        Generating adjacent cells if not yet visited.
        """

        if direction is None:  # counterclockwise
            functions = (
                self.cell_upright,
                self.cell_up,
                self.cell_left,
                self.cell_downleft,
                self.cell_down,
                self.cell_right,
            )
        elif direction:  # optimized for white
            functions = (
                self.cell_down,
                self.cell_right,
                self.cell_upright,
                self.cell_up,
                self.cell_downleft,
                self.cell_left,
            )
        else:  # optimized for black
            functions = (
                self.cell_right,
                self.cell_down,
                self.cell_downleft,
                self.cell_left,
                self.cell_upright,
                self.cell_up,
            )

        among: BitBoard = ~self.occupied & ~visited if empty_only else ~visited
        for f in functions:
            try:
                neighbour_cell = f(mask)
            except BoardShapeError:
                continue
            else:
                if neighbour_cell & among:
                    yield neighbour_cell

    def generate_local_moves(
        self, last_move_mask: BitBoard
    ) -> Generator[BitBoard, None, None]:
        """
        Generate moves directly adjacent to the last move.
        """

        for adjacent_mask in self.generate_adjacent_cells(
            last_move_mask, visited=0, empty_only=True
        ):
            yield adjacent_mask

    def generate_bridge_moves(
        self, visited: BitBoard = 0
    ) -> Generator[BitBoard, None, None]:
        """
        Generate moves that create bridges with own stones.
        """

        functions = (
            self.bridge_black_up,
            self.bridge_black_down,
            self.bridge_white_right,
            self.bridge_white_left,
            self.bridge_diag_right,
            self.bridge_diag_left,
            self.cell_up,
        )

        empty_not_visited = self.unoccupied & ~visited
        occupied_generator = (
            self.generate_white_occupied()
            if self.turn
            else self.generate_black_occupied()
        )
        for mask in occupied_generator:
            for f in functions:
                try:
                    neighbour_cell = f(mask)
                except BoardShapeError:
                    continue
                else:
                    if neighbour_cell & empty_not_visited:
                        yield neighbour_cell

    @property
    def legal_moves(self) -> Generator[BitBoard, None, None]:
        """"""

        return self.generate_moves()

    def generate_moves(self) -> Generator[BitBoard, None, None]:
        """
        Generating all moves, but adjacent (local) first.
        """

        visited: BitBoard = yield from self.generate_adjacent_moves()

        yield from self.generate_remaining_moves(visited)

    def generate_adjacent_moves(
        self, visited: BitBoard = 0
    ) -> Generator[BitBoard, None, BitBoard]:
        """"""

        for mask in self.generate_occupied():
            for adjacent_mask in self.generate_adjacent_cells(
                mask, visited, empty_only=True
            ):
                yield adjacent_mask
                visited |= adjacent_mask

        return visited

    def generate_occupied(self) -> Generator[BitBoard, None, None]:
        """"""

        if self.turn:
            yield from self.generate_black_occupied()
            yield from self.generate_white_occupied()
        else:
            yield from self.generate_white_occupied()
            yield from self.generate_black_occupied()

    def generate_black_occupied(self) -> Generator[BitBoard, None, None]:
        """
        Generating from first byte, that is from top-left to bottom-right on the hex board.
        """

        yield from generate_masks(self.occupied_co[False])

    def generate_white_occupied(self) -> Generator[BitBoard, None, None]:
        """"""

        yield from generate_masks(self.occupied_co[True])

    def generate_remaining_moves(
        self, visited: BitBoard
    ) -> Generator[BitBoard, None, None]:
        """"""

        yield from generate_masks(self.unoccupied & ~visited)

    def position(self) -> str:
        """"""

    def copy(self) -> HexBoard:
        """"""

    def push_coord(self, coord: str) -> None:
        """"""

        self.push(Move.from_cord(coord, self.size).mask)

    def push(self, move: BitBoard) -> None:
        """"""

        self.occupied_co[self.turn] |= move

        self.move_stack.append(Move(move))

        self.turn = not self.turn

    def pop(self) -> None:
        """"""

        self.occupied_co[not self.turn] &= ~self.move_stack.pop().mask

        self.turn = not self.turn

    def get_forcing_level(self, move: BitBoard) -> int:
        """
        Get how forcing is the hypothetical move (should be empty at the moment).

        Considered forcing when adjacent to both own and opponent stone *or* when adjacent to lone opponent stone.
        """

        adjacent_black: Optional[BitBoard] = None
        adjacent_white: Optional[BitBoard] = None

        # adjacent to both colors
        for mask in self.generate_adjacent_cells(move):
            if not adjacent_black and mask & self.occupied_co[False]:
                adjacent_black = mask
            if not adjacent_white and mask & self.occupied_co[True]:
                adjacent_white = mask

            if adjacent_black and adjacent_white:
                return 1

        # adjacent to a lone stone
        if self.turn and adjacent_black:
            if not any(self.generate_adjacent_cells(adjacent_black, empty_only=True)):
                return 1
        elif not self.turn and adjacent_white:
            if not any(self.generate_adjacent_cells(adjacent_white, empty_only=True)):
                return 1

        return 0

    def is_check(self) -> bool:
        """
        Get if is check.
        TODO: do something about it (should not be required in base class)
        """

        return False

    def get_connectedness_and_wingspan(self) -> Tuple[int, int]:
        """
        Scan in every of 3 dimensions of the board, in each aggregating series of stones of same color.

        If no stone of opposition color is in-between then stones are considered "connected" and their wingspan is
        equal to distance between them.
        """

        cw_counter: CWCounter = CWCounter(0, 0, 0, 0)

        self._loop_all_cells(cw_counter, lambda mask, i: mask << 1)
        self._loop_all_cells(
            cw_counter,
            lambda mask, i: mask << self.size
            if i % self.size == self.size - 1
            else mask >> (self.size - 1),
        )
        self._loop_all_cells(
            cw_counter,
            lambda mask, i: mask << (self.size - 1)
            if self._x(mask) != 0
            else mask >> (self._y(mask) * (self.size - 1) - 1),
        )

        return (
            cw_counter.white_connectedness - cw_counter.black_connectedness,
            cw_counter.white_wingspan - cw_counter.black_wingspan,
        )

    def _loop_all_cells(self, cw_counter: CWCounter, mask_shift: Callable) -> None:
        """"""

        mask: int = 1
        last_occupied: Optional[bool] = None
        wingspan_counter: int = 0

        # iterate over each column, moving cell by cell from left to right
        for i in range(self.size**2):
            if i % self.size == 0:
                wingspan_counter = 0
                last_occupied = None

            else:
                if mask & self.occupied_co[False]:
                    if last_occupied is False:
                        cw_counter.black_connectedness += 1
                        cw_counter.black_wingspan += wingspan_counter

                    last_occupied = False
                    wingspan_counter = 0

                elif mask & self.occupied_co[True]:
                    if last_occupied is True:
                        cw_counter.white_connectedness += 1
                        cw_counter.white_wingspan += wingspan_counter

                    last_occupied = True
                    wingspan_counter = 0

            mask = mask_shift(mask)
            wingspan_counter += 1

    def get_imbalance(self, color: bool) -> float:
        """
        Sum up if stones are distributed in a balanced way across:
            - left/right
            - top/left
            - center/edge
        """

        half_size: float = self.size / 2

        occupied = self.occupied_co[color]

        xs: List[int] = []
        ys: List[int] = []
        center_distances: List[float] = []

        for mask in generate_masks(occupied):
            x = self._x(mask)
            y = self._y(mask)
            xs.append(x)
            ys.append(y)
            center_distances.append((half_size - x) ** 2 + (half_size - y) ** 2)

        imbalance_x = abs(half_size - mean(xs))
        imbalance_y = abs(half_size - mean(ys))
        imbalance_center = abs((self.size / 4) ** 2 - mean(center_distances))

        return imbalance_x + imbalance_y + imbalance_center

    def _x(self, mask: BitBoard) -> int:
        """"""

        return (mask - 1).bit_length() % self.size

    def _y(self, mask: BitBoard) -> int:
        """"""

        return mask.bit_length() // self.size
