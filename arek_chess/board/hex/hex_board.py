# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from copy import copy
from dataclasses import dataclass
from functools import reduce
from itertools import groupby, permutations, product
from operator import ior
from typing import Callable, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from nptyping import Int8, NDArray, Shape
from numpy import empty, float32, int8, mean, zeros

from arek_chess.board import BitBoard, GameBoardBase
from arek_chess.board.hex.mixins import BoardShapeError
from arek_chess.board.hex.mixins.bitboard_utils import generate_masks
from arek_chess.board.hex.mixins.hex_board_serializer_mixin import (
    HexBoardSerializerMixin,
)
from arek_chess.common.constants import DEFAULT_HEX_BOARD_SIZE
from astar import find_path

Cell = int

SIZE = 13

VEC_1: BitBoard = 2**13 - 1
NEIGHBOURHOOD_DIAMETER: int = 7
ZERO: int8 = int8(0)
ONE: int8 = int8(1)
TWO: int8 = int8(2)


@dataclass
class CWCounter:
    black_connectedness: int
    white_connectedness: int
    black_wingspan: int
    white_wingspan: int


@dataclass
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


class HexBoard(HexBoardSerializerMixin, GameBoardBase):
    """
    Handling the hex board and calculating features of a position.
    """

    turn: bool
    """The side to move True for white, False for black."""

    move_stack: List[Move]
    """List of moves on board from first to last."""

    occupied_co: Dict[bool, BitBoard]
    unoccupied: BitBoard

    has_move_limit: bool = True

    initial_notation: str

    def __init__(
        self,
        notation: Optional[str] = None,
        *,
        size: int = DEFAULT_HEX_BOARD_SIZE,
        init_move_stack: bool = False,
    ) -> None:
        """"""

        super().__init__()

        self.initial_notation = notation
        self.size = size
        self.size_square = size**2
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
        self.reset()

        if notation:
            self.initialize_notation(notation, init_move_stack)

        self.short_diagonal_mask = self._get_short_diagonal_mask()
        self.long_diagonal_mask = self._get_long_diagonal_mask()

    def initialize_notation(self, notation: str, init_move_stack: bool = False) -> None:
        """"""

        move_str: str = ""
        color: bool = False
        for is_letter, value in groupby(notation, str.isalpha):
            move_str += "".join(value)
            if not is_letter:
                mask = Move.mask_from_coord(move_str, self.size)
                self.occupied_co[color] |= mask
                if init_move_stack:
                    self.move_stack.append(Move(mask, self.size))

                move_str = ""
                color = not color

        self.unoccupied ^= self.occupied_co[True] | self.occupied_co[False]

        self.turn = color

    def get_notation(self) -> str:
        """"""

        notation: str = self.initial_notation or ""

        for move in self.move_stack:
            notation += move.get_coord()

        return notation

    def reset(self) -> None:
        """"""

        self.turn = False
        self.occupied_co = {False: 0, True: 0}
        self.unoccupied = self.get_all_mask()
        self.move_stack = []

    def get_all_mask(self) -> BitBoard:
        """"""

        return (1 << self.size_square) - 1

    def color_at(self, mask: BitBoard) -> Optional[bool]:
        """"""

        return self.color_at_mask(mask)

    def mask_at(self, x: int, y: int) -> BitBoard:
        """"""

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

    def is_adjacent(self, mask_1: BitBoard, mask_2: BitBoard):
        """"""

        if mask_1 is mask_2:
            return False

        return self.is_adjacent_mask(mask_1, mask_2)

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
        among: Optional[BitBoard] = None,
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

        for f in functions:
            try:
                neighbour_cell = f(mask)
            except BoardShapeError:
                continue
            else:
                if (among is not None and neighbour_cell & among) or (
                    among is None and neighbour_cell
                ):
                    yield neighbour_cell

    def generate_local_moves(
        self, last_move_mask: BitBoard
    ) -> Generator[BitBoard, None, None]:
        """
        Generate moves directly adjacent to the last move.
        """

        for adjacent_mask in self.generate_adjacent_cells(
            last_move_mask, among=self.unoccupied
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
    def legal_moves(self) -> Generator[Move, None, None]:
        """"""

        if self.winner() is not None:
            return self.generate_nothing()

        return self.generate_moves()

    @staticmethod
    def generate_nothing() -> Generator[Move, None, None]:
        """"""

        yield from ()

    def generate_moves(self) -> Generator[Move, None, None]:
        """
        Generating all moves, but adjacent (local) first.
        """

        visited: BitBoard = yield from self.generate_adjacent_moves()

        visited = yield from self.generate_diagonal_moves(visited)

        yield from self.generate_remaining_moves(visited)

    def generate_adjacent_moves(
        self, visited: BitBoard = 0
    ) -> Generator[Move, None, BitBoard]:
        """"""

        for mask in self.generate_occupied():
            for adjacent_mask in self.generate_adjacent_cells(
                mask, among=self.unoccupied & ~visited
            ):
                yield Move(adjacent_mask, self.size)
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

    def generate_diagonal_moves(
        self, visited: BitBoard
    ) -> Generator[Move, None, BitBoard]:
        """"""

        to_visit: BitBoard = (
            (self.short_diagonal_mask | self.long_diagonal_mask)
            & self.unoccupied
            & ~visited
        )
        for mask in generate_masks(to_visit):
            yield Move(mask, self.size)

        return visited | to_visit

    def generate_remaining_moves(
        self, visited: BitBoard
    ) -> Generator[Move, None, None]:
        """"""

        yield from (
            Move(mask, self.size) for mask in generate_masks(self.unoccupied & ~visited)
        )

    def position(self) -> str:
        """"""

        return self.get_notation()  # TODO: was it meant to return bitboard maybe?

    def copy(self) -> HexBoard:
        """"""

        return copy(self)

    def push_coord(self, coord: str) -> None:
        """"""

        self.push(Move.from_coord(coord, self.size))

    def push(self, move: Move) -> None:
        """"""

        if move.mask & self.unoccupied == 0:
            print("board", bin(self.unoccupied), bin(self.occupied))
            raise ValueError(f"the move is occupied: {move.uci()}")

        self.occupied_co[self.turn] |= move.mask
        self.unoccupied ^= move.mask

        self.move_stack.append(move)

        self.turn = not self.turn

    def pop(self) -> Move:
        """"""

        move = self.move_stack.pop()

        self.occupied_co[not self.turn] ^= move.mask
        self.unoccupied |= move.mask
        self.turn = not self.turn

        return move

    def get_forcing_level(self, move: Move) -> int:
        """
        Get how forcing is the move.

        Considered forcing when adjacent to both own and opponent stone *or* when adjacent to lone opponent stone.
        """

        return 0

        adjacent_black: Optional[BitBoard] = None
        adjacent_white: Optional[BitBoard] = None

        # adjacent to both colors
        for mask in self.generate_adjacent_cells(move.mask):
            if not adjacent_black and mask & self.occupied_co[False]:
                adjacent_black = mask
            if not adjacent_white and mask & self.occupied_co[True]:
                adjacent_white = mask

            if adjacent_black and adjacent_white:
                return 1

        # adjacent to a lone stone
        if self.turn and adjacent_black:
            if not any(
                self.generate_adjacent_cells(adjacent_black, among=self.unoccupied)
            ):
                return 1
        elif not self.turn and adjacent_white:
            if not any(
                self.generate_adjacent_cells(adjacent_white, among=self.unoccupied)
            ):
                return 1

        return 0

    def is_check(self) -> bool:
        """
        Get if is check.
        TODO: do something about it (should not be required in base class),
         !!! currently always True to indicate no draw in evaluator !!!
        """

        return True

    def get_connectedness_and_wingspan(self) -> Tuple[int, int, int, int]:
        """
        Scan in every of 3 dimensions of the board, in each aggregating series of stones of same color.

        If no stone of opposition color is in-between then stones are considered "connected" and their wingspan is
        equal to distance between them.
        """

        cw_counter: CWCounter = CWCounter(0, 0, 0, 0)

        self._walk_all_cells(
            cw_counter, True, lambda mask, i: mask << 1
        )  # left to right
        self._walk_all_cells(  # top to bottom
            cw_counter,
            False,
            lambda mask, i: (
                y := self._y(mask),
                mask << self.size
                if y != self.size - 1
                else mask >> self.size * (self.size - 1) - 1,
            )[1],
        )
        self._walk_all_cells(  # top-right to bottom-left, along short diagonal
            cw_counter,
            None,
            lambda mask, i: (
                x := self._x(mask),
                y := self._y(mask),
                mask << 1
                if mask == 1 or i >= self.size_square - 2
                else mask << (self.size - 1)
                if x != 0 and y != self.size - 1
                else mask >> (y * (self.size - 1) - 1)
                if x == 0 and y != self.size - 1
                else mask >> (self.size * (self.size - x - 2) - (self.size - x - 1)),
            )[2],
        )

        return (
            cw_counter.white_connectedness,
            cw_counter.black_connectedness,
            cw_counter.white_wingspan,
            cw_counter.black_wingspan,
        )

    def _walk_all_cells(
        self, cw_counter: CWCounter, edge_color: Optional[bool], mask_shift: Callable
    ) -> None:
        """
        Walk over all cells using given shift from cell to cell.

        :param cw_counter: connectedness and wingspan counter object
        :param edge_color: color of the edge (point on start and finish)
        :param mask_shift: function that shifts the mask on every iteration
        """

        mask: int = 1
        last_occupied: Optional[bool] = None
        wingspan_counter: int = 0
        ec: Optional[bool]
        """Edge color at the end of iteration."""

        # iterate over each column, moving cell by cell from left to right
        for i in range(self.size_square):
            x = self._x(mask)
            y = self._y(mask)
            if (edge_color is not None and i % self.size == 0) or (
                edge_color is None and (x == self.size - 1 or y == 0)
            ):
                ec = self._increment_on_finished_column(
                    cw_counter, edge_color, i, wingspan_counter, last_occupied
                )

                wingspan_counter = 1
                last_occupied = (
                    edge_color if edge_color is not None else not ec
                )  # first edge opposite to final edge

            if mask & self.occupied_co[False]:
                if not last_occupied:  # None or False
                    cw_counter.black_connectedness += 1
                    cw_counter.black_wingspan += wingspan_counter

                last_occupied = False
                wingspan_counter = 0

            elif mask & self.occupied_co[True]:
                if last_occupied is None or last_occupied is True:
                    cw_counter.white_connectedness += 1
                    cw_counter.white_wingspan += wingspan_counter

                last_occupied = True
                wingspan_counter = 0

            mask = mask_shift(mask, i)
            wingspan_counter += 1

        self._increment_on_finished_column(
            cw_counter, edge_color, i, wingspan_counter, last_occupied
        )

    def _increment_on_finished_column(
        self,
        cw_counter: CWCounter,
        edge_color: Optional[bool],
        i: int,
        wingspan_counter: int,
        last_occupied: Optional[bool],
    ) -> Optional[bool]:
        """"""

        ec = edge_color if edge_color is not None else self._get_final_edge_color(i)

        # increment counters for the connection with the edge at the end of iteration (first iteration excluded)
        if i != 0 and wingspan_counter <= self.size:
            if last_occupied is False and not ec:  # None or False
                cw_counter.black_connectedness += 1
                cw_counter.black_wingspan += wingspan_counter
            if last_occupied is True and (ec is None or ec is True):
                cw_counter.white_connectedness += 1
                cw_counter.white_wingspan += wingspan_counter

        return ec

    def _get_final_edge_color(self, i: int) -> Optional[bool]:
        """
        Get closing edge of the board on last cell of column/row.

        This returns the edge color for iterating along short diagonal from top-right towards bottom-left.

        :param i: iteration counter

        :returns: None on short diagonal, white in the top-left triangle-half of the board, otherwise black.
        """

        half_number_of_cells: int = (self.size_square - self.size) // 2
        """Half of cells except the short diagonal."""

        if i <= half_number_of_cells:
            return True
        elif i <= half_number_of_cells + self.size:
            return None
        else:
            return False

    def get_imbalance(self, color: bool) -> Tuple[float32, float32]:
        """
        Sum up if stones are distributed in a balanced way across:
            - left/right
            - top/left
            - center/edge
        """

        half_size: float = self.size / 2 - 0.5

        occupied = self.occupied_co[color]
        if not occupied:
            return float32(0), float32(0)

        xs: List[int] = []
        ys: List[int] = []
        center_distances: List[float] = []

        for mask in generate_masks(occupied):
            x = self._x(mask)
            y = self._y(mask)
            xs.append(x)
            ys.append(y)
            center_distances.append((half_size - x) ** 2 + (half_size - y) ** 2)

        imbalance_x = float32(abs(half_size - mean(xs)))
        imbalance_y = float32(abs(half_size - mean(ys)))
        imbalance_center = float32(
            abs((self.size / 4) - math.sqrt(mean(center_distances)))
        )

        return imbalance_x + imbalance_y, imbalance_center

    def get_neighbourhood(
        self, diameter: int = NEIGHBOURHOOD_DIAMETER, should_suppress: bool = False
    ) -> NDArray[Shape, Int8]:
        """
        Return a collection of states of cells around the cell that was played last.

        If the move is on the border then

        The order in the collection should be subsequent rows top to bottom, each row from left to right.
        """

        if self.size < diameter:
            raise ValueError("Cannot get neighbourhood larger than board size")

        shift = (diameter - 1) // 2

        if not self.occupied_co[True] | self.occupied_co[False]:
            return zeros((diameter ** 2,), dtype=int8)

        if not self.move_stack:
            if should_suppress:
                return zeros((diameter ** 2,), dtype=int8)

            raise ValueError("Cannot get neighbourhood of `None`")
            # print("empty board")

        move: Move = self.move_stack[-1]
        move_x: int = move.x
        move_y: int = move.y

        top_left_mask_x = (
            move_x - shift
            if shift <= move_x <= self.size - shift - 1
            else 0
            if move_x < shift
            else self.size - 1
        )
        top_left_mask_y = (
            move_y - shift
            if shift <= move_y <= self.size - shift - 1
            else 0
            if move_y < shift
            else self.size - 1
        )

        mask: BitBoard = Move.mask_from_xy(top_left_mask_x, top_left_mask_y, self.size)
        """Top-left mask at start, then shifted."""

        array = empty((diameter, diameter), dtype=int8)
        for row in range(diameter):
            for col in range(diameter):
                occupied_white = mask & self.occupied_co[True]
                occupied_black = mask & self.occupied_co[False]
                array[row][col] = (
                    TWO if occupied_black else ONE if occupied_white else ZERO
                )

                if col != diameter - 1:  # last iteration
                    mask <<= 1

            if row != diameter - 1:  # last iteration
                mask <<= self.size - diameter + 1

        return array.flatten()

    def _x(self, mask: BitBoard) -> int:
        """"""

        return Move._x(mask, self.size)

    def _y(self, mask: BitBoard) -> int:
        """"""

        return Move._y(mask, self.size)

    def _get_short_diagonal_mask(self) -> BitBoard:
        """"""

        mask = 0

        pointer = 1 << (self.size - 1)
        for i in range(self.size):
            mask |= pointer
            if i < self.size - 1:
                pointer = self.cell_downleft(pointer)

        pointer = 1 << (self.size - 1)
        for i in range(self.size - 1):
            mask |= pointer
            if i < self.size - 2:
                pointer = self.cell_downleft(pointer)

        pointer = 1 << (2 * self.size - 1)
        for i in range(self.size - 1):
            mask |= pointer
            if i < self.size - 2:
                pointer = self.cell_downleft(pointer)

        return mask

    def _get_long_diagonal_mask(self) -> BitBoard:
        """"""

        mask = 0

        pointer = 1
        for i in range(self.size):
            mask |= pointer
            if i < self.size - 1:
                pointer = self.bridge_diag_right(pointer)

        pointer = 1 << 1
        for i in range(self.size - 1):
            mask |= pointer
            if i < self.size - 2:
                pointer = self.bridge_diag_right(pointer)

        pointer = 1 << self.size
        for i in range(self.size - 1):
            mask |= pointer
            if i < self.size - 2:
                pointer = self.bridge_diag_right(pointer)

        return mask

    def get_move_limit(self) -> int:
        """
        :returns: number of permutations how the board can be filled
        """

        unoccupied_count = self.unoccupied.bit_count()
        return math.factorial(unoccupied_count)

    def distance(self, m1: BitBoard, m2: BitBoard, color: bool) -> int:
        """
        Calculate distance from one cell to another, assuming that stepping on opposition stones is prohibited.
        """

        path = find_path(
            m1,
            m2,
            neighbors_fnct=lambda m: self.generate_adjacent_cells(
                m, among=self.unoccupied | self.occupied_co[color]
            ),
        )
        if path is None:
            raise ValueError("Path does not exist between the given points")
        return len(path) - 1

    def distance_missing(self, m1: BitBoard, m2: BitBoard, color: bool) -> int:
        """
        Calculate how many stones are missing to finish the connection from one cell to another.
        """

        path = find_path(
            m1,
            m2,
            neighbors_fnct=lambda m: self.generate_adjacent_cells(
                m, among=self.unoccupied | self.occupied_co[color]
            ),
            distance_between_fnct=lambda m1, m2: 1
            if self.unoccupied
            & (m1 | m2)  # additional cost so that it avoids stepping on empty
            else 0,
            heuristic_cost_estimate_fnct=lambda a, b: 0,
        )

        if path is None:
            return self.size

        return len([mask for mask in path if mask & self.unoccupied])

    def get_shortest_missing_distance(self, color: bool) -> int:
        """
        Calculate how many stones are missing to finish the connection between two sides.
        """

        connection_points_start: List[BitBoard] = self._get_start_points(color)
        connection_points_finish: List[BitBoard] = self._get_finish_points(color)

        if not (connection_points_start and connection_points_finish):
            raise ValueError("searching shortest missing distance on game over")

        connection_points_pairs: product[Tuple[BitBoard, BitBoard]] = product(
            connection_points_start, connection_points_finish
        )
        return min([self.distance_missing(*pair, color) for pair in connection_points_pairs])

    def _get_start_points(self, color: bool) -> List[BitBoard]:
        """"""

        own = self.occupied_co[color]

        if color:
            # all own cells at first column
            points = own & self.bb_cols[0]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_cols[0] & ~opp))
                return [masks[len(masks)//2]]  # take a single point in the middle of an edge

        else:
            # all own cells at first row
            points = own & self.bb_rows[0]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_rows[0] & ~opp))
                return [masks[len(masks)//2]]  # take a single point in the middle of an edge

    def _get_finish_points(self, color: bool) -> List[BitBoard]:
        """"""

        own = self.occupied_co[color]

        if color:
            # all own cells at first column
            points = own & self.bb_cols[-1]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_cols[-1] & ~opp))
                return [masks[len(masks)//2]]  # take a single point in the middle of an edge

        else:
            # all own cells at first row
            points = own & self.bb_rows[-1]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_rows[-1] & ~opp))
                return [masks[len(masks)//2]]  # take a single point in the middle of an edge
