# -*- coding: utf-8 -*-
from __future__ import annotations

import math
from collections import defaultdict
from copy import copy
from functools import reduce, lru_cache
from itertools import groupby, product
from operator import ior
from random import randint
from typing import (
    Callable,
    Generator,
    Iterator,
    Optional,
)

import torch as th
from astar import find_path
from nptyping import Shape
import numpy as np
from numpy import asarray, empty, int8, mean, zeros, ndarray
from torch_geometric.data import Data as GraphData, HeteroData

from hackable_engine.board import BitBoard, GameBoardBase
from hackable_engine.board.hex.bitboard_utils import (
    generate_masks,
    int_to_inverse_binary_array,
    get_random_mask,
)
from hackable_engine.board.hex.move import Move, CWCounter
from hackable_engine.board.hex.serializers import BoardShapeError
from hackable_engine.board.hex.serializers.hex_board_serializer_mixin import (
    HexBoardSerializerMixin,
)
from hackable_engine.board.hex.types import EdgeType
from hackable_engine.common.constants import DEFAULT_HEX_BOARD_SIZE, FLOAT_TYPE

Cell = int

SIZE = 13
SIZE_SQUARE = SIZE**2

NEIGHBOURHOOD_DIAMETER: int = 7
ZERO = 0
ONE = 1
TWO = 2
MINUS_ONE = -1


class HexBoard(HexBoardSerializerMixin, GameBoardBase):
    """
    Handling the hex board and calculating features of a position.
    """

    has_draws = False

    turn: bool
    """The side to move True for white, False for black."""

    move_stack: list[Move]
    """list of moves on board from first to last."""

    occupied_co: dict[bool, BitBoard]
    unoccupied: BitBoard

    has_move_limit: bool = True

    initial_notation: str

    def __init__(
        self,
        notation: str = "",
        *,
        size: int = DEFAULT_HEX_BOARD_SIZE,
        init_move_stack: bool = False,
        use_graph: bool = False,
    ) -> None:
        """"""

        super().__init__()

        self.initial_notation = notation if not init_move_stack else ""
        self.size = size
        self.size_square = size**2
        self.half_number_of_cells: int = (self.size_square - self.size) // 2
        """Half of cells except the short diagonal."""

        self.bb_rows: list[BitBoard] = [
            reduce(ior, self._generate_row_masks(row)) for row in range(size)
        ]
        self.bb_cols: list[BitBoard] = [
            reduce(ior, self._generate_col_masks(col)) for col in range(size)
        ]
        self.vertical_coeff = 2**self.size
        self.diagonal_coeff = 2 ** (self.size - 1)
        self.reset()

        if notation:
            self.initialize_notation(notation, init_move_stack)

        if use_graph:
            # self.edge_index = th.tensor(list(self._get_all_graph_links()), dtype=th.long).t().contiguous()
            self.edge_index = self._get_all_graph_links_coo()
            self.edge_types = self._get_graph_link_types()
        else:
            self.edge_index = th.tensor([])
            self.edge_types = th.tensor([])

        # self.short_diagonal_mask = self._get_short_diagonal_mask()
        # self.long_diagonal_mask = self._get_long_diagonal_mask()

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

    def set_move_stack_from_notation(self, notation: str) -> None:
        """"""

        move_str: str = ""
        for is_letter, value in groupby(notation, str.isalpha):
            move_str += "".join(value)
            if not is_letter:
                mask = Move.mask_from_coord(move_str, self.size)
                self.move_stack.append(Move(mask, self.size))

                move_str = ""

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

        return Move.from_xy(x, y, self.size).mask

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

    @lru_cache(SIZE_SQUARE)
    def cell_right(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first column on the left
        if mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift right {bin(mask)}")

        # cells are ordered opposite direction to bits in bitboard
        return mask << 1

    @lru_cache(SIZE_SQUARE)
    def cell_left(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first column on the left
        if mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift left {bin(mask)}")

        # cells are ordered opposite direction to bits in bitboard
        return mask >> 1

    @lru_cache(SIZE_SQUARE)
    def cell_down(self, mask: BitBoard) -> BitBoard:
        """"""

        # 0 is the first row on top
        if mask & self.bb_rows[self.size - 1]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask << self.size

    @lru_cache(SIZE_SQUARE)
    def cell_up(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask >> self.size

    @lru_cache(SIZE_SQUARE)
    def cell_downleft(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (self.size - 1)

    @lru_cache(SIZE_SQUARE)
    def cell_upright(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift down {bin(mask)}")

        return mask >> (self.size - 1)

    @lru_cache(SIZE_SQUARE)
    def bridge_diag_right(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (self.size + 1)

    @lru_cache(SIZE_SQUARE)
    def bridge_diag_left(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[0] or mask & self.bb_cols[0]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (self.size + 1)

    @lru_cache(SIZE_SQUARE)
    def bridge_black_down(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask << (2 * self.size - 1)

    @lru_cache(SIZE_SQUARE)
    def bridge_black_up(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (2 * self.size - 1)

    @lru_cache(SIZE_SQUARE)
    def bridge_white_right(self, mask: BitBoard) -> BitBoard:
        """"""

        if mask & self.bb_rows[self.size - 1] or mask & self.bb_cols[self.size - 1]:
            raise BoardShapeError(f"trying to shift downleft {bin(mask)}")

        return mask >> (self.size - 2)

    @lru_cache(SIZE_SQUARE)
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

    def winner_no_turn(self) -> Optional[bool]:
        """"""

        if self.is_black_win():  # last move was black
            return False

        if self.is_white_win():  # last move was white
            return True

        return None

    def is_black_win(self) -> bool:
        """"""

        blacks: BitBoard = self.occupied_co[False]

        # check there is a black stone on every row
        if not all((blacks & row for row in self.bb_rows)):
            return False

        # find connection from top to bottom
        visited = 0
        for mask in generate_masks(blacks & self.bb_rows[0]):
            finished, visited = self.is_connected_to_bottom(mask, visited)
            if finished:
                return True

        return False

    def is_connected_to_bottom(
        self, mask: BitBoard, visited: BitBoard = 0
    ) -> tuple[bool, int]:
        """
        Recurrent way of finding if a stone is connected to bottom.
        """

        if mask & self.bb_rows[self.size - 1]:
            return True, visited

        visited |= mask

        for neighbour in self.generate_neighbours_black(mask, visited):
            finished, visited = self.is_connected_to_bottom(neighbour, visited)
            if finished:
                return True, visited

        return False, visited

    def generate_neighbours(self, mask: BitBoard) -> Iterator:
        """
        Generate masks that are neighbours of a given mask.
        """

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
                yield neighbour_cell

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
        visited = 0
        for mask in generate_masks(whites & self.bb_cols[0]):
            finished, visited = self.is_connected_to_right(mask, visited)
            if finished:
                return True

        return False

    def is_connected_to_right(
        self, mask: BitBoard, visited: BitBoard = 0
    ) -> tuple[bool, int]:
        """
        Recurrent way of finding if a stone is connected to bottom.
        """

        if mask & self.bb_cols[self.size - 1]:
            return True, visited

        visited |= mask

        for neighbour in self.generate_neighbours_white(mask, visited):
            finished, visited = self.is_connected_to_right(neighbour, visited)
            if finished:
                return True, visited

        return False, visited

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

        functions: tuple[Callable[[BitBoard], BitBoard], ...]
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

    def generate_adjacent_cells_cached(
        self,
        mask: BitBoard,
        among: BitBoard,
        direction: Optional[bool] = None,
    ) -> Generator[BitBoard, None, None]:
        """
        Generating adjacent cells if not yet visited.
        """

        yield from generate_masks(
            self._generate_adjacent_cells_cached(mask, direction) & among
        )

    @lru_cache(SIZE_SQUARE)
    def _generate_adjacent_cells_cached(
        self, mask: BitBoard, direction: bool | None
    ) -> BitBoard:
        adjacent_cells: BitBoard = 0

        functions: tuple[Callable[[BitBoard], BitBoard], ...]
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
                adjacent_cells |= f(mask)
            except BoardShapeError:
                continue

        return adjacent_cells

    def generate_local_moves(
        self, last_move_mask: BitBoard
    ) -> Generator[BitBoard, None, None]:
        """
        Generate moves directly adjacent to the last move.
        """

        yield from self.generate_adjacent_cells(last_move_mask, among=self.unoccupied)

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
        """
        Returns a generator of all legal moves, that is, all empty cells on the board.

        It is not considered if the game is over, but only which cells are occupied.
        """

        # if self.winner() is not None:
        #     return self.generate_nothing()

        return self.generate_moves()

    @staticmethod
    def generate_nothing() -> Generator[Move, None, None]:
        """"""

        yield from ()

    def get_random_move(self) -> Move:
        """"""

        return Move(
            self.get_random_unoccupied_mask(self.size_square - len(self.move_stack)),
            self.size,
        )

    def get_random_unoccupied_mask(self, empty_cells: Optional[int] = None) -> BitBoard:
        """"""

        if empty_cells == 0:
            raise ValueError("no unoccupied cells in a fully filled board")

        empty_cells = empty_cells or self.size_square

        return get_random_mask(self.unoccupied, randint(0, empty_cells - 1))

    def generate_moves(self) -> Generator[Move, None, None]:
        """
        Generating all moves, but adjacent (local) first.
        """

        visited: BitBoard = yield from self.generate_adjacent_moves()

        # visited = yield from self.generate_diagonal_moves(visited)

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

    # def generate_diagonal_moves(
    #     self, visited: BitBoard
    # ) -> Generator[Move, None, BitBoard]:
    #     """"""
    #
    #     to_visit: BitBoard = (
    #         (self.short_diagonal_mask | self.long_diagonal_mask)
    #         & self.unoccupied
    #         & ~visited
    #     )
    #     for mask in generate_masks(to_visit):
    #         yield Move(mask, self.size)
    #
    #     return visited | to_visit

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
            # print("board", bin(self.unoccupied), bin(self.occupied))
            # print(self.move_stack)
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

        adjacent_black: Optional[BitBoard] = None
        adjacent_white: Optional[BitBoard] = None

        # adjacent to both colors
        for mask in self.generate_adjacent_cells(move.mask):
            if not adjacent_black and mask & self.occupied_co[False]:
                adjacent_black = mask
            if not adjacent_white and mask & self.occupied_co[True]:
                adjacent_white = mask

            # if adjacent_black and adjacent_white:
            #     return 1

        # adjacent to a lone stone
        if self.turn and adjacent_black:
            if not any(
                self.generate_adjacent_cells(adjacent_black, among=self.unoccupied)
            ):
                return 1  # TODO: should be 2 if possible to identify nozoki
            return 1
        elif not self.turn and adjacent_white:
            if not any(
                self.generate_adjacent_cells(adjacent_white, among=self.unoccupied)
            ):
                return 1
            return 1

        return 0

    def is_check(self) -> bool:
        """
        Get if is check.
        TODO: do something about it (should not be required in base class),
         currently always True to indicate no draw in evaluator
        """

        return True

    def get_connectedness_and_spacing(self) -> tuple[int, int, int, int]:
        """
        Scan in every of 3 dimensions of the board, in each aggregating series of stones of same color.

        If no stone of opposition color is in-between then stones are considered "connected" and their spacing is
        equal to distance between them.
        """

        cw_counter: CWCounter = CWCounter(0, 0, 0, 0)

        self._walk_all_cells(  # left to right
            cw_counter, True, lambda mask, i: mask << 1
        )
        self._walk_all_cells(  # top to bottom
            cw_counter,
            False,
            lambda mask, i: (
                y := self._y(mask),
                (
                    mask << self.size
                    if y != self.size - 1
                    else mask >> self.size * (self.size - 1) - 1
                ),
            )[1],
        )
        self._walk_all_cells(  # top-right to bottom-left, along short diagonal
            cw_counter,
            None,
            lambda mask, i: (
                x := self._x(mask),
                y := self._y(mask),
                (
                    mask << 1
                    if mask == 1 or i >= self.size_square - 2
                    else (
                        mask << (self.size - 1)
                        if x != 0 and y != self.size - 1
                        else (
                            mask >> (y * (self.size - 1) - 1)
                            if x == 0 and y != self.size - 1
                            else mask
                            >> (self.size * (self.size - x - 2) - (self.size - x - 1))
                        )
                    )
                ),
            )[2],
        )

        return (
            cw_counter.white_connectedness,
            cw_counter.black_connectedness,
            cw_counter.white_spacing,
            cw_counter.black_spacing,
        )

    def _walk_all_cells(
        self, cw_counter: CWCounter, edge_color: Optional[bool], mask_shift: Callable
    ) -> None:
        """
        Walk over all cells using given shift from cell to cell.

        :param cw_counter: connectedness and spacing counter object
        :param edge_color: color of the edge (point on start and finish)
        :param mask_shift: function that shifts the mask on every iteration
        """

        mask: int = 1
        last_occupied: Optional[bool] = None
        spacing_counter: int = 0
        ec: Optional[bool]
        """Edge color at the end of iteration."""

        size = self.size
        oc_black = self.occupied_co[False]
        oc_white = self.occupied_co[True]

        # iterate over each column, moving cell by cell from left to right
        i: int = 0
        for i in range(self.size_square):
            x = (mask.bit_length() - 1) % size
            y = (mask.bit_length() - 1) // size

            if (edge_color is not None and i % size == 0) or (
                edge_color is None and (x == size - 1 or y == 0)
            ):
                ec = self._increment_on_finished_column(
                    cw_counter, edge_color, i, spacing_counter, last_occupied
                )

                spacing_counter = 1
                last_occupied = (
                    edge_color if edge_color is not None else not ec
                )  # first edge opposite to final edge

            if mask & oc_black:
                if not last_occupied:  # None or False
                    cw_counter.black_connectedness += 1
                    cw_counter.black_spacing += spacing_counter

                last_occupied = False
                spacing_counter = 0

            elif mask & oc_white:
                if last_occupied is None or last_occupied is True:
                    cw_counter.white_connectedness += 1
                    cw_counter.white_spacing += spacing_counter

                last_occupied = True
                spacing_counter = 0

            mask = mask_shift(mask, i)
            spacing_counter += 1

        self._increment_on_finished_column(
            cw_counter, edge_color, i, spacing_counter, last_occupied
        )

    def _increment_on_finished_column(
        self,
        cw_counter: CWCounter,
        edge_color: Optional[bool],
        i: int,
        spacing_counter: int,
        last_occupied: Optional[bool],
    ) -> Optional[bool]:
        """"""

        ec = edge_color if edge_color is not None else self._get_final_edge_color(i)

        # increment counters for the connection with the edge at the end of iteration (first iteration excluded)
        if i != 0 and spacing_counter <= self.size:
            if last_occupied is False and not ec:  # None or False
                cw_counter.black_connectedness += 1
                cw_counter.black_spacing += spacing_counter
            if last_occupied is True and (ec is None or ec is True):
                cw_counter.white_connectedness += 1
                cw_counter.white_spacing += spacing_counter

        return ec

    def _get_final_edge_color(self, i: int) -> Optional[bool]:
        """
        Get closing edge of the board on last cell of column/row.

        This returns the edge color for iterating along short diagonal from top-right towards bottom-left.

        :param i: iteration counter

        :returns: None on short diagonal, white in the top-left triangle-half of the board, otherwise black.
        """

        if i <= self.half_number_of_cells:
            return True
        elif i <= self.half_number_of_cells + self.size:
            return None
        else:
            return False

    def get_imbalance(self, color: bool) -> tuple[FLOAT_TYPE, FLOAT_TYPE]:
        """
        Sum up if stones are distributed in a balanced way across:
            - left/right
            - top/left
            - center/edge

        Values theoretically within:
            first between 0 and `board.size` / 2
            second between 0 and `board.size` / 4
        """

        half_size: float = self.size / 2 - 0.5

        occupied = self.occupied_co[color]
        if not occupied:
            return FLOAT_TYPE(0), FLOAT_TYPE(0)

        xs: list[int] = []
        ys: list[int] = []
        center_distances: list[float] = []

        for mask in generate_masks(occupied):
            x = self._x(mask)
            y = self._y(mask)
            xs.append(x)
            ys.append(y)
            center_distances.append((half_size - x) ** 2 + (half_size - y) ** 2)

        imbalance_x = FLOAT_TYPE(abs(half_size - mean(xs)))
        imbalance_y = FLOAT_TYPE(abs(half_size - mean(ys)))
        imbalance_center = FLOAT_TYPE(
            abs((self.size / 4) - math.sqrt(mean(center_distances)))
        )

        return imbalance_x + imbalance_y, imbalance_center

    def get_neighbourhood(
        self, diameter: int = NEIGHBOURHOOD_DIAMETER, should_suppress: bool = False
    ) -> ndarray[Shape, int8]:
        """
        Return a collection of states of cells around the cell that was played last.

        If the move is on the border then

        The order in the collection should be subsequent rows top to bottom, each row from left to right.

        :return: array of shape (self.size, self.size)
        """

        if self.size < diameter:
            raise ValueError("Cannot get neighbourhood larger than board size")

        radius = (diameter - 1) // 2
        """Included cells around the last move, in straight line."""

        if not self.occupied_co[True] | self.occupied_co[False]:
            return zeros((diameter**2,), dtype=int8)

        if not self.move_stack:
            if should_suppress:
                return zeros((diameter**2,), dtype=int8)

            raise ValueError("Cannot get neighbourhood of `None`")
            # print("empty board")

        move: Move = self.move_stack[-1]
        move_x: int = move.x
        move_y: int = move.y

        top_left_x = (
            0
            if self.size == radius or move_x <= radius
            else (
                self.size - diameter
                if move_x >= self.size - radius
                else move_x - radius
            )
        )
        top_left_y = (
            0
            if self.size == radius or move_y <= radius
            else (
                self.size - diameter
                if move_y >= self.size - radius
                else move_y - radius
            )
        )

        mask: BitBoard = Move.mask_from_xy(top_left_x, top_left_y, self.size)
        """Top-left mask at start, then assigned subsequent masks."""

        array = empty((diameter, diameter), dtype=int8)
        for row in range(diameter):
            for col in range(diameter):
                occupied_white = mask & self.occupied_co[True]
                occupied_black = mask & self.occupied_co[False]
                array[row][col] = (
                    MINUS_ONE if occupied_black else ONE if occupied_white else ZERO
                )

                if col != diameter - 1:  # not last iteration
                    mask <<= 1

            if row != diameter - 1:  # not last iteration
                mask <<= self.size - diameter + 1

        return array

    def as_matrix(
        self, black_stone_val: FLOAT_TYPE = MINUS_ONE, empty_val: FLOAT_TYPE = ZERO
    ) -> ndarray[Shape, FLOAT_TYPE]:
        """"""

        return self._as_matrix(
            self.size, self.size_square, self.occupied_co[True], self.occupied_co[False]
        )
        # array: ndarray[Shape, FLOAT_TYPE] = empty((1, self.size, self.size), dtype=FLOAT_TYPE)
        #
        # return self._as_matrix(
        #     array,
        #     self.size,
        #     self.occupied_co[True],
        #     self.occupied_co[False],
        #     black_stone_val,
        #     empty_val,
        # )

    def as_matrix_legacy(
        self, black_stone_val: FLOAT_TYPE = MINUS_ONE, empty_val: FLOAT_TYPE = ZERO
    ) -> ndarray[Shape, FLOAT_TYPE]:
        array: ndarray[Shape, FLOAT_TYPE] = empty((1, self.size, self.size), dtype=FLOAT_TYPE)
        return self._as_matrix_to_array(
            array,
            self.size,
            self.occupied_co[True],
            self.occupied_co[False],
            black_stone_val,
            empty_val,
        )

    # @numba.njit()
    @staticmethod
    def _as_matrix_to_array(
        array: ndarray[Shape, FLOAT_TYPE],
        size: int,
        occupied_white: int,
        occupied_black: int,
        black_val: FLOAT_TYPE = MINUS_ONE,
        empty_val: FLOAT_TYPE = ZERO,
    ) -> ndarray[Shape, FLOAT_TYPE]:
        """"""

        mask: BitBoard = 1

        for row in range(size):
            for col in range(size):
                array[0][row][col] = (
                    black_val
                    if mask & occupied_black
                    else ONE if mask & occupied_white else empty_val
                )

                mask <<= 1

        return array

    @staticmethod
    def _as_matrix(
        size: int,
        size_square: int,
        occupied_white: int,
        occupied_black: int,
        black_val: FLOAT_TYPE = MINUS_ONE,
        empty_val: FLOAT_TYPE = ZERO,
    ) -> ndarray[Shape, FLOAT_TYPE]:
        """"""

        mask: BitBoard = 1
        array: list[BitBoard] = []

        for i in range(size_square):
            if mask & occupied_black:
                value = black_val
            elif mask & occupied_white:
                value = ONE
            else:
                value = empty_val
            array.append(value)
            mask <<= 1

        return np.array(array, dtype=FLOAT_TYPE).reshape((1, size, size))

    def as_matrix_channelled(self) -> ndarray:
        """"""

        array: ndarray = empty((2, self.size, self.size), dtype=int8)

        return self._as_matrix_channelled(
            array,
            self.size,
            self.occupied_co[True],
            self.occupied_co[False],
        )

    @staticmethod
    def _as_matrix_channelled(
        array: ndarray[Shape, int8],
        size: int,
        occupied_white: int,
        occupied_black: int,
    ) -> ndarray:
        """"""

        mask: BitBoard = 1

        for row in range(size):
            for col in range(size):
                if occupied_white & mask:
                    array[1][row][col] = 1
                elif occupied_black & mask:
                    array[0][row][col] = 1

                mask <<= 1

        return array

    def _as_matrix_efficiency(
        self,
        array: ndarray[Shape, int8],
        black_val: int8 = int8(MINUS_ONE),
        empty_val: int8 = int8(ZERO),
    ) -> ndarray:
        """"""

        o_white = int_to_inverse_binary_array(self.occupied_co[True], self.size_square)
        o_black = int_to_inverse_binary_array(self.occupied_co[False], self.size_square)
        return self._as_matrix_numba(
            array, self.size, o_white, o_black, black_val, empty_val
        )

    @staticmethod
    # @numba.njit()
    def _as_matrix_numba(
        array: ndarray[Shape, int8],
        size: int,
        o_white: ndarray[Shape, int8],
        o_black: ndarray[Shape, int8],
        black_val: int8 = int8(MINUS_ONE),
        empty_val: int8 = int8(ZERO),
    ) -> ndarray:
        """"""

        pos = size**2 - 1

        for row in range(size):
            for col in range(size):
                occupied_white = bool(o_white[pos])
                occupied_black = bool(o_black[pos])
                array[row][col] = (
                    black_val
                    if occupied_black
                    else ONE if occupied_white else empty_val
                )

                pos -= 1

        return array

    def _x(self, mask: BitBoard) -> int:
        """"""

        return Move.get_x(mask, self.size)

    def _y(self, mask: BitBoard) -> int:
        """"""

        return Move.get_y(mask, self.size)

    def _get_short_diagonal_mask(self) -> BitBoard:
        """
        FIXME: this is not working right, for instance at 3x3 is showing 5 cells
        """

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

    def _generate_row_masks(self, row: int) -> Generator[BitBoard, None, None]:
        """"""

        yield from (1 << (col + row * self.size) for col in range(self.size))

    def _generate_col_masks(self, col: int) -> Generator[BitBoard, None, None]:
        """"""

        yield from ((1 << col) << (row * self.size) for row in range(self.size))

    def get_area_moves(
        self, col_range: tuple[int, int], row_range: tuple[int, int]
    ) -> list[Move]:
        """"""

        return [
            Move(mask, self.size)
            for mask in generate_masks(self.get_area_mask(col_range, row_range))
        ]

    def get_area_mask(
        self, col_range: tuple[int, int], row_range: tuple[int, int]
    ) -> BitBoard:
        """"""

        return (
            reduce(ior, self.generate_masks_from_area(col_range, row_range))
            & self.occupied
        )

    def generate_masks_from_area(
        self, col_range: tuple[int, int], row_range: tuple[int, int]
    ) -> Generator[BitBoard, None, None]:
        """"""

        yield from (
            1 << (col + row * self.size)
            for col in range(*col_range)
            for row in range(*row_range)
        )

    def generate_moves_from_area(
        self,
        col_range: tuple[int, int],
        row_range: tuple[int, int],
        visited: BitBoard = 0,
    ) -> Generator[Move, None, None]:
        """"""

        for mask in self.generate_masks_from_area(col_range, row_range):
            if mask & self.unoccupied & ~visited:
                yield Move(mask, self.size)

    def get_move_limit(self) -> int:
        """
        :returns: number of permutations how the board can be filled
        """

        unoccupied_count = self.unoccupied.bit_count()
        return math.factorial(unoccupied_count)

    def distance_missing(
        self, mask_from: BitBoard, mask_to: BitBoard, color: bool
    ) -> tuple[int, list[BitBoard]]:
        """
        Calculate how many stones are missing to finish the connection from one cell to another.

        The astar algorithm is imperfect, a wrong result example is written in tests for this module.
        """

        un_oc = self.unoccupied
        oc_co = self.occupied_co[color]
        gen = self.generate_adjacent_cells_cached
        among = un_oc | oc_co

        path = find_path(
            mask_from,
            mask_to,
            neighbors_fnct=lambda mask: gen(mask, among=among),
            distance_between_fnct=lambda m_from, m_to: (0.0 if m_to & oc_co else 1.0),
        )

        if path is None:
            return (self.size_square, [])

        masks = [mask for mask in path if mask & self.unoccupied]
        return len(masks), masks

    @lru_cache(maxsize=10_000_000)
    def distance_missing_cached(
        self,
        mask_from: BitBoard,
        mask_to: BitBoard,
        color: bool,
        un_oc: BitBoard,
        oc_co: BitBoard,
    ) -> tuple[int, list[BitBoard]]:
        """
        Calculate how many stones are missing to finish the connection from one cell to another.

        The astar algorithm is imperfect, a wrong result example is written in tests for this module.
        """

        gen = self.generate_adjacent_cells_cached
        among = un_oc | oc_co

        path = find_path(
            mask_from,
            mask_to,
            neighbors_fnct=lambda mask: gen(mask, among=among),
            distance_between_fnct=lambda m_from, m_to: (0.0 if m_to & oc_co else 1.0),
        )

        if path is None:
            return (self.size_square, [])

        masks = [mask for mask in path if mask & self.unoccupied]
        return len(masks), masks

    def get_shortest_missing_distance(self, color: bool) -> int:
        """
        Calculate how many stones are missing to finish the connection between two sides.
        """

        connection_points_start: list[BitBoard] = self._get_start_points(color)
        connection_points_finish: list[BitBoard] = self._get_finish_points(color)

        if not (connection_points_start and connection_points_finish):
            raise ValueError("searching shortest missing distance on game over")

        connection_points_pairs: product[tuple[BitBoard, BitBoard]] = product(
            connection_points_start, connection_points_finish
        )
        return min(
            d
            for d, p in (
                self.distance_missing(*pair, color) for pair in connection_points_pairs
            )
        )

    def get_short_missing_distances(
        self, color: bool, *, should_subtract: bool = False
    ) -> tuple[int, dict[float, int]]:
        """
        Calculate how many stones are missing to finish the connection between two sides.

        :param should_subtract: when evaluating distances for white side with 1 stone less on board,
            subtracts 0.5, because an additional stone cannot have impact on all the variants
        """

        connection_points_start: list[BitBoard] = self._get_start_points(color)
        connection_points_finish: list[BitBoard] = self._get_finish_points(color)

        if not (connection_points_start and connection_points_finish):
            raise ValueError("searching shortest missing distance on game over")

        connection_points_pairs: product[tuple[BitBoard, BitBoard]] = product(
            connection_points_start, connection_points_finish
        )

        shortest_distance = self.size_square
        variants: dict[float, int] = defaultdict(lambda: 0)
        unique_paths: set[tuple[BitBoard, ...]] = set()
        shortest_path: set[BitBoard] = set()

        un_oc = self.unoccupied
        for pair in connection_points_pairs:
            path: list[BitBoard]

            if un_oc.bit_count() >= self.size_square - 2 * self.size:
                oc_co = self.occupied_co[color]
                length, path = self.distance_missing_cached(*pair, color, un_oc, oc_co)
            else:
                length, path = self.distance_missing(*pair, color)
            if not path:
                continue

            if length < shortest_distance:
                shortest_distance = length
                shortest_path = set(path)
                unique_paths = {
                    p for p in unique_paths if not shortest_path.issubset(p)
                }

            masks = tuple(path)
            if (
                length < self.size
                and masks not in unique_paths
                and (
                    length == shortest_distance
                    or not shortest_path.issubset(set(masks))
                )
            ):
                unique_paths.add(masks)

        for p in unique_paths:
            variants[len(p) if not should_subtract else len(p) - 0.5] += 1

        return shortest_distance, variants

    def get_short_missing_distances_perf(
        self, color: bool, *, should_subtract: bool = False
    ) -> tuple[int, dict[float, int]]:
        """
        Calculate how many stones are missing to finish the connection between two sides.
        Search restricted to connections from/to the obtuse corners.

        :param should_subtract: when evaluating distances for white side with 1 stone less on board,
            subtracts 0.5, because an additional stone cannot have impact on all the variants
        """

        opp = self.occupied_co[not color]
        if color:
            start_corner = list(generate_masks(self.bb_cols[0] & ~opp))[-1]
            finish_corner = next(generate_masks(self.bb_cols[-1] & ~opp))
        else:
            start_corner = list(generate_masks(self.bb_rows[0] & ~opp))[-1]
            finish_corner = next(generate_masks(self.bb_rows[-1] & ~opp))

        connection_points_start: list[BitBoard] = self._get_start_points_perf(color)
        connection_points_finish: list[BitBoard] = self._get_finish_points_perf(color)

        if not (connection_points_start and connection_points_finish):
            raise ValueError("searching shortest missing distance on game over")

        connection_points_pairs: list[tuple[BitBoard, BitBoard]] = [
            (start_corner, finish) for finish in connection_points_finish
        ] + [(start, finish_corner) for start in connection_points_start]

        shortest_distance = self.size_square
        variants: dict[float, int] = defaultdict(lambda: 0)
        unique_paths: set[tuple[BitBoard, ...]] = set()
        shortest_path: set[BitBoard] = set()

        un_oc = self.unoccupied

        for pair in connection_points_pairs:
            path: list[BitBoard]
            if un_oc.bit_count() >= self.size_square - self.size:
                oc_co = self.occupied_co[color]
                length, path = self.distance_missing_cached(*pair, color, un_oc, oc_co)
            else:
                length, path = self.distance_missing(*pair, color)
            if not path:
                continue

            if length < shortest_distance:
                shortest_distance = length
                shortest_path = set(path)
                unique_paths = {
                    p for p in unique_paths if not shortest_path.issubset(p)
                }

            masks = tuple(path)
            if (
                length < self.size
                and masks not in unique_paths
                and (
                    length == shortest_distance
                    or not shortest_path.issubset(set(masks))
                )
            ):
                unique_paths.add(masks)

        for p in unique_paths:
            variants[len(p) if not should_subtract else len(p) - 0.5] += 1

        return shortest_distance, variants

    def pair_name(self, pair):
        """"""

        a = Move(pair[0], size=self.size).get_coord()
        b = Move(pair[1], size=self.size).get_coord()
        return f"{a},{b}"

    def get_shortest_missing_distance_perf(self, color: bool) -> int:
        """
        Calculate how many stones are missing to finish the connection between two sides.

        Measurements restricted to: from obtuse corner to any opposite cell.
        """

        opp = self.occupied_co[not color]
        if color:
            start_corner = list(generate_masks(self.bb_cols[0] & ~opp))[-1]
            finish_corner = next(generate_masks(self.bb_cols[-1] & ~opp))
        else:
            start_corner = list(generate_masks(self.bb_rows[0] & ~opp))[-1]
            finish_corner = next(generate_masks(self.bb_rows[-1] & ~opp))

        connection_points_start: list[BitBoard] = self._get_start_points_perf(color)
        connection_points_finish: list[BitBoard] = self._get_finish_points_perf(color)

        if not (connection_points_start and connection_points_finish):
            raise ValueError("searching shortest missing distance on game over")

        connection_points_pairs: list[tuple[BitBoard, BitBoard]] = [
            (start_corner, finish) for finish in connection_points_finish
        ] + [(start, finish_corner) for start in connection_points_start]

        return min(
            (
                distance
                for distance, path in (
                    self.distance_missing(*pair, color)
                    for pair in connection_points_pairs
                )
            )
        )

    def _get_start_points(self, color: bool) -> list[BitBoard]:
        """Top row or left column."""

        opp = self.occupied_co[not color]
        if color:
            masks = list(generate_masks(self.bb_cols[0] & ~opp))
            return masks

        else:
            masks = list(generate_masks(self.bb_rows[0] & ~opp))
            return masks

    def _get_start_points_perf(self, color: bool) -> list[BitBoard]:
        """Top row or left column."""

        own = self.occupied_co[color]

        if color:
            # all own cells at first column
            points = own & self.bb_cols[0]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_cols[0] & ~opp))
                return masks
                # return [masks[len(masks)//2]]  # take a single point in the middle of an edge

        else:
            # all own cells at first row
            points = own & self.bb_rows[0]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_rows[0] & ~opp))
                return masks
                # return [masks[len(masks)//2]]  # take a single point in the middle of an edge

    def _get_finish_points(self, color: bool) -> list[BitBoard]:
        """Bottom row or right column."""

        if color:
            opp = self.occupied_co[not color]
            masks = list(generate_masks(self.bb_cols[-1] & ~opp))
            return masks

        else:
            opp = self.occupied_co[not color]
            masks = list(generate_masks(self.bb_rows[-1] & ~opp))
            return masks

    def _get_finish_points_perf(self, color: bool) -> list[BitBoard]:
        """Bottom row or right column."""

        own = self.occupied_co[color]

        if color:
            # all own cells at last column
            points = own & self.bb_cols[-1]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_cols[-1] & ~opp))
                return masks
                # return [masks[len(masks)//2]]  # take a single point in the middle of an edge

        else:
            # all own cells at last row
            points = own & self.bb_rows[-1]
            if points:
                return list(generate_masks(points))
            else:
                opp = self.occupied_co[not color]
                masks = list(generate_masks(self.bb_rows[-1] & ~opp))
                return masks
                # return [masks[len(masks)//2]]  # take a single point in the middle of an edge

    def color_matrix(self, color: bool) -> ndarray:
        """"""

        array = zeros((self.size, self.size), dtype=int)

        for mask in generate_masks(self.occupied_co[color]):
            x, y = Move.xy_from_mask(mask, self.size)
            array[x][y] = 1

        return array

    def _get_all_graph_links_coo(self) -> th.Tensor:
        """
        Get `edge_index` for graph data.
        """

        return (
            th.tensor(
                [
                    (link[0].bit_length() - 1, link[1].bit_length() - 1)
                    for link in self._get_all_graph_links()
                ],
                dtype=th.long,
            )
            .t()
            .contiguous()
        )

    def _get_all_graph_links(self) -> set[tuple[BitBoard, BitBoard]]:
        """
        Return all links between all board cells, considering the board to be a graph.
        """

        links: set[tuple[BitBoard, BitBoard]] = set()

        for mask in generate_masks(self.get_all_mask()):
            for neighbour_mask in self.generate_neighbours(mask):
                links.add((mask, neighbour_mask))
                # links.add((neighbour_mask, mask))  # this should always be added anyway within the outer loop

        return links

    def _get_graph_link_types(self) -> th.Tensor:
        """Returns one-hot encoded edge types as tensor with shape (num_edges, 3)"""
        link_types: list[EdgeType] = []
        link: tuple[BitBoard, BitBoard]
        for from_, to_ in self.edge_index.t():
            from_x, from_y = Move.xy_from_mask(from_.item(), self.size)
            to_x, to_y = Move.xy_from_mask(to_.item(), self.size)
            if from_x == to_x:
                link_types.append(EdgeType.VERTICAL)
            elif from_y == to_y:
                link_types.append(EdgeType.HORIZONTAL)
            else:
                link_types.append(EdgeType.DIAGONAL)
        return th.tensor(link_types)

    def _get_graph_link_types_one_hot(self) -> th.Tensor:
        """Returns one-hot encoded edge types as tensor with shape (num_edges, 3)"""
        link_types = []
        link: tuple[BitBoard, BitBoard]
        for from_, to_ in self.edge_index.t():
            from_x, from_y = Move.xy_from_mask(from_.item(), self.size)
            to_x, to_y = Move.xy_from_mask(to_.item(), self.size)
            if from_x == to_x:
                link_types.append([1, 0, 0])  # EdgeType.VERTICAL)
            elif from_y == to_y:
                link_types.append([0, 1, 0])  # EdgeType.HORIZONTAL)
            else:
                link_types.append([0, 0, 1])  # EdgeType.DIAGONAL)
        return th.tensor(link_types)

    def to_homo_graph_data(self) -> GraphData:
        """"""

        return GraphData(
            x=th.from_numpy(self.get_homo_graph_node_features()),
            edge_index=self.edge_index,
        )

    def get_homo_graph_node_features(self) -> ndarray:
        """
        Get node features, where the only feature is stone color (or lack thereof).
        :return: tensor of shape (self.size_square, 1)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for _ in range(self.size_square):
            node_feature = 0
            if whites & 1:
                node_feature = 1
            elif blacks & 1:
                node_feature = -1

            node_features.append(node_feature)

            whites >>= 1
            blacks >>= 1

        return np.array([node_features], dtype=FLOAT_TYPE).transpose()

    def to_hetero_graph_data(self) -> HeteroData:
        """"""

        return HeteroData(
            x=th.from_numpy(self.get_hetero_graph_node_features()),
            edge_index=self.edge_index,
        )

    def get_hetero_graph_node_features(self) -> ndarray:
        """
        Get node features, where the features are: stone color, edge of the board.
        :return: tensor of shape (self.size_square, 1)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for i in range(self.size_square):
            row = i // self.size
            col = i % self.size
            stone = -1.0 if blacks & 1 else 1.0 if whites & 1 else 0.0
            edge_white = -1.0 if col == 0 else 1.0 if col == (self.size - 1) else 0.0
            edge_black = -1.0 if row == 0 else 1.0 if row == (self.size - 1) else 0.0

            node_features.append((stone, edge_black, edge_white))

            whites >>= 1
            blacks >>= 1

        return np.array([node_features], dtype=FLOAT_TYPE).transpose()

    def get_hetero_graph_node_embedding(self) -> ndarray:
        """
        Get node features, where the features are: stone color, edge of the board.
        :return: tensor of shape (self.size_square, 1)
        """

        node_features = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        for i in range(self.size_square):
            row = i // self.size
            col = i % self.size
            stone = 0 if blacks & 1 else 1 if whites & 1 else 2
            edge_white = 0 if col == 0 else 1 if col == (self.size - 1) else 2
            edge_black = 0 if row == 0 else 1 if row == (self.size - 1) else 2

            node_features.append((stone, edge_black, edge_white))

            whites >>= 1
            blacks >>= 1

        return np.array(node_features, dtype=FLOAT_TYPE).transpose()

    def _get_nodes_and_links(self) -> tuple[ndarray, ndarray]:
        """
        By convention the board graph has always all the nodes and link types, counting empty.

        Only the type of the node and link changes.
            - Available node types: empty, white, black.
            - Available link types: empty, white, black, mixed.

        To obtain a full graph a third value is required, which is constant per board - table of links.

        :returns: node types and link types
        """

        nodes = []
        links = []

        whites = self.occupied_co[True]
        blacks = self.occupied_co[False]

        last_color: int = 0
        for i in range(self.size_square):
            color: int = 0
            if whites & 1:
                color = 1
            elif blacks & 1:
                color = -1

            nodes.append(color)
            if i % self.size != 0:
                links.append(self._get_graph_link_type(last_color, color))

            last_color = color

            whites >>= 1
            blacks >>= 1

        self._append_graph_vertical_links_(nodes, links)
        self._append_graph_diagonal_links_(nodes, links)

        return asarray(nodes), asarray(links)

    def _append_graph_vertical_links_(self, nodes: list[int], links: list[int]):
        """
        Links along columns.

        TODO: think if better done same as `_append_graph_diagonal_links`
        """

        last_color: int = 0
        for k in range(self.size):
            for i in range(k, self.size_square + k, self.size):
                color = nodes[i]
                if i >= self.size:
                    # links along rows
                    links.append(self._get_graph_link_type(last_color, color))
                last_color = color

    def _append_graph_diagonal_links_(self, nodes: list[int], links: list[int]):
        """
        Links along the short diagonal.
        """

        for i in range(self.size_square):
            color = nodes[i]
            try:
                neighbour_mask = self.cell_downleft(1 << i)
                if not color or self.unoccupied & neighbour_mask:
                    links.append(0)

                elif color == 1 and self.occupied_co[True] & neighbour_mask:
                    links.append(1)
                elif color == 2 and self.occupied_co[False] & neighbour_mask:
                    links.append(2)
                else:
                    links.append(3)

            except BoardShapeError:
                continue

    @staticmethod
    def _get_graph_link_type(last_color: Optional[int], color: int) -> int:
        """"""

        if not last_color or not color:
            return 0

        elif last_color == 1 and color == 1:
            return 1

        elif last_color == 2 and color == 2:
            return 2

        return 3
