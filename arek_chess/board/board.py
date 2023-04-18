# -*- coding: utf-8 -*-
"""
Handling the chessboard and calculating features of a position for RL training observation.
"""

from __future__ import annotations

import collections
import sys
from functools import reduce
from typing import (
    List,
    Optional,
    Counter,
    Hashable,
    cast,
    Iterator,
    Literal,
    Union,
    Dict,
    Tuple,
    Iterable,
)

from chess import (
    Board as ChessBoard,
    Bitboard,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    square_rank,
    Square,
    Color,
    BB_KING_ATTACKS,
    BB_KNIGHT_ATTACKS,
    BB_RANK_ATTACKS,
    BB_FILE_ATTACKS,
    BB_DIAG_ATTACKS,
    BB_PAWN_ATTACKS,
    BB_LIGHT_SQUARES,
    BB_DARK_SQUARES,
    Move,
    Outcome,
    Termination,
    BoardT,
    _BoardState,
    square_file,
    BB_DIAG_MASKS,
    BB_RANK_MASKS,
    BB_FILE_MASKS,
    PieceType,
    square_distance, scan_reversed, BB_ALL, BB_RANK_3, BB_RANK_4, BB_RANK_5, BB_RANK_6,
)
from nptyping import NDArray, Shape, Int, Double
from numpy import double, empty

SQUARES = [
    A1,
    B1,
    C1,
    D1,
    E1,
    F1,
    G1,
    H1,
    A2,
    B2,
    C2,
    D2,
    E2,
    F2,
    G2,
    H2,
    A3,
    B3,
    C3,
    D3,
    E3,
    F3,
    G3,
    H3,
    A4,
    B4,
    C4,
    D4,
    E4,
    F4,
    G4,
    H4,
    A5,
    B5,
    C5,
    D5,
    E5,
    F5,
    G5,
    H5,
    A6,
    B6,
    C6,
    D6,
    E6,
    F6,
    G6,
    H6,
    A7,
    B7,
    C7,
    D7,
    E7,
    F7,
    G7,
    H7,
    A8,
    B8,
    C8,
    D8,
    E8,
    F8,
    G8,
    H8,
] = range(64)
BB_SQUARES = [
    BB_A1,
    BB_B1,
    BB_C1,
    BB_D1,
    BB_E1,
    BB_F1,
    BB_G1,
    BB_H1,
    BB_A2,
    BB_B2,
    BB_C2,
    BB_D2,
    BB_E2,
    BB_F2,
    BB_G2,
    BB_H2,
    BB_A3,
    BB_B3,
    BB_C3,
    BB_D3,
    BB_E3,
    BB_F3,
    BB_G3,
    BB_H3,
    BB_A4,
    BB_B4,
    BB_C4,
    BB_D4,
    BB_E4,
    BB_F4,
    BB_G4,
    BB_H4,
    BB_A5,
    BB_B5,
    BB_C5,
    BB_D5,
    BB_E5,
    BB_F5,
    BB_G5,
    BB_H5,
    BB_A6,
    BB_B6,
    BB_C6,
    BB_D6,
    BB_E6,
    BB_F6,
    BB_G6,
    BB_H6,
    BB_A7,
    BB_B7,
    BB_C7,
    BB_D7,
    BB_E7,
    BB_F7,
    BB_G7,
    BB_H7,
    BB_A8,
    BB_B8,
    BB_C8,
    BB_D8,
    BB_E8,
    BB_F8,
    BB_G8,
    BB_H8,
] = [1 << sq for sq in SQUARES]

FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]
PIECE_SYMBOLS = [None, "p", "n", "b", "r", "q", "k"]
SQUARE_NAMES = [f + r for r in RANK_NAMES for f in FILE_NAMES]


def scan_forward(bb: Bitboard) -> Iterator[Square]:
    while bb:
        r = bb & -bb
        yield r.bit_length() - 1
        bb ^= r


if sys.version_info[1] <= 9:
    from gmpy2.gmpy2 import popcount  # type: ignore

    def get_bit_count(bb: Bitboard) -> int:
        # return bin(bb).count("1")
        return popcount(bb)

else:

    def get_bit_count(bb: Bitboard) -> int:
        # return bin(bb).count("1")
        return bb.bit_count()


class Board(ChessBoard):
    """
    Handling the chessboard and calculating features of a position.
    """

    _stack: List[_BoardState]

    def as_nonunique_int(self) -> int:
        """"""

        black, white = self.occupied_co

        return sum(
            (
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
                self.kings,
                black,
                white,
            )
        )

    def as_unique_int(self) -> int:
        """"""

        black, white = self.occupied_co

        return reduce(
            lambda s, e: (s << 64 | e),
            (
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
                self.kings,
                black,
                white,
            ),
        )

    @classmethod
    def from_unique_int(cls, position: int) -> Board:
        """"""

        b = cls()

        white, black, b.kings, b.queens, b.rooks, b.bishops, b.knights, b.pawns = [
            ONES & (position >> k * 64) for k in range(8)
        ]
        b.occupied_co = [black, white]
        b.occupied = black | white

        return b

    def get_material_simple(self, color: bool) -> int:
        return (
            get_bit_count(self.pieces_mask(piece_type=PAWN, color=color))
            + 3 * get_bit_count(self.pieces_mask(piece_type=KNIGHT, color=color))
            + 3 * get_bit_count(self.pieces_mask(piece_type=BISHOP, color=color))
            + 5 * get_bit_count(self.pieces_mask(piece_type=ROOK, color=color))
            + 9 * get_bit_count(self.pieces_mask(piece_type=QUEEN, color=color))
        )

    def get_material_simple_both(self) -> int:
        return (
            get_bit_count(self.pawns)
            + 3 * get_bit_count(self.knights)
            + 3 * get_bit_count(self.bishops)
            + 5 * get_bit_count(self.rooks)
            + 9 * get_bit_count(self.queens)
        )

    def get_pawns_simple_both(self) -> int:
        return get_bit_count(self.pawns)

    def get_material_pawns(self, color: bool) -> double:
        pawn_bb: Bitboard = self.pawns & self.occupied_co[color]

        value: double = double(0.0)
        for bb in scan_forward(pawn_bb):
            value += self.get_pawn_value(square_rank(bb), color)

        return value

    @staticmethod
    def get_pawn_value(rank: int, color: bool) -> double:
        if not color:
            rank = 7 - rank

        return PAWN_VALUES[rank]

    def get_piece_value(
        self, color: Color, piece_type: Optional[PieceType], rank: int = 0
    ) -> double:
        if piece_type is None or piece_type == 6:
            return double(0)
        elif piece_type == 1:
            return self.get_pawn_value(rank, color) if rank else double(1)
        elif piece_type in [2, 3]:
            return double(3)
        elif piece_type == 4:
            return double(5)
        elif piece_type == 5:
            return double(9)
        return double(0)

    def get_material_no_pawns(self, color: bool) -> int:
        oc_co: Bitboard = self.occupied_co[color]

        return (
            3 * get_bit_count(self.knights & oc_co)
            + 3 * get_bit_count(self.bishops & oc_co)
            + 5 * get_bit_count(self.rooks & oc_co)
            + 9 * get_bit_count(self.queens & oc_co)
        )

    def get_material_no_pawns_both(self) -> int:
        return (
            3 * get_bit_count(self.knights)
            + 3 * get_bit_count(self.bishops)
            + 5 * get_bit_count(self.rooks)
            + 9 * get_bit_count(self.queens)
        )

    def mask_to_square(self, mask: Bitboard) -> Square:
        return BB_SQUARES.index(mask)

    def mask_to_squares(self, mask: Bitboard) -> List[Square]:
        squares = []
        for square in scan_forward(mask):
            squares.append(square)
        return squares

    def attacks_mask(self, square: Square) -> Bitboard:
        bb_square = BB_SQUARES[square]

        if bb_square & self.pawns:
            color = bool(bb_square & self.occupied_co[True])
            return BB_PAWN_ATTACKS[color][square]
        elif bb_square & self.knights:
            return BB_KNIGHT_ATTACKS[square]
        elif bb_square & self.kings:
            return BB_KING_ATTACKS[square]
        else:
            attacks = 0
            if bb_square & self.bishops or bb_square & self.queens:
                attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
            if bb_square & self.rooks or bb_square & self.queens:
                attacks |= (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
                )
            return attacks

    def threats_mask(self, color: Color, square: Square) -> Bitboard:
        queens_and_rooks = self.queens | self.rooks
        queens_and_bishops = self.queens | self.bishops

        attackers = (
            (BB_KING_ATTACKS[square] & self.kings)
            | (BB_KNIGHT_ATTACKS[square] & self.knights)
            | (BB_RANK_ATTACKS[square][0] & queens_and_rooks)
            | (BB_FILE_ATTACKS[square][0] & queens_and_rooks)
            | (BB_DIAG_ATTACKS[square][0] & queens_and_bishops)
            | (BB_PAWN_ATTACKS[not color][square] & self.pawns)
        )

        return attackers & self.occupied_co[color]

    def direct_threats_mask(self, color: Color, square: Square) -> Bitboard:
        """Attacks of knights and pawns"""

        attackers = (BB_KNIGHT_ATTACKS[square] & self.knights) | (
            BB_PAWN_ATTACKS[not color][square] & self.pawns
        )

        return attackers & self.occupied_co[color]

    def get_king_mobility(self, color: Color) -> int:
        """Own pieces/pawns and enemy pawns are considered shield."""

        shield = self.occupied_co[color] | (self.occupied_co[not color] & self.pawns)

        return get_bit_count(
            KING_ATTACKS[self.kings & self.occupied_co[color]] & ~shield
        )

    def get_protection(self, color: Color) -> int:
        """"""

        protection = 0

        for square in scan_forward(self.occupied_co[color] & ~self.kings):
            protection += self._has_attacker(color, square)

        return protection

    def _has_attacker(
        self, color: Color, square: Square
    ) -> Union[Literal[0], Literal[1]]:
        oc_co: Bitboard = self.occupied_co[color]

        if BB_KING_ATTACKS[square] & self.kings & oc_co:
            return 1

        if BB_KNIGHT_ATTACKS[square] & self.knights & oc_co:
            return 1

        if BB_PAWN_ATTACKS[not color][square] & self.pawns & oc_co:
            return 1

        oc: Bitboard = self.occupied
        rank_at: Bitboard = BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & oc]
        if rank_at & self.rooks & oc_co:
            return 1
        elif rank_at & self.queens & oc_co:
            return 1

        file_at: Bitboard = BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & oc]
        if file_at & self.rooks & oc_co:
            return 1
        elif file_at & self.queens & oc_co:
            return 1

        diag_at: Bitboard = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & oc]
        if diag_at & self.bishops & oc_co:
            return 1
        elif diag_at & self.queens & oc_co:
            return 1

        return 0

    def get_square_control_map_for_both(self) -> NDArray[Shape["64"], Int]:
        """
        Returns list of 64 values, accumulated attacks on each square.
        """

        pawns: Bitboard = self.pawns
        kings: Bitboard = self.kings
        knights: Bitboard = self.knights
        q_and_r: Bitboard = self.queens | self.rooks
        q_and_b: Bitboard = self.queens | self.bishops
        oc = self.occupied
        oc_co_white: Bitboard = self.occupied_co[True]
        oc_co_black: Bitboard = self.occupied_co[False]

        arr = empty(shape=(64,), dtype=int)
        _attackers_mask_light_for_both = self._attackers_mask_light_for_both
        for (
            square,
            king_att,
            knight_att,
            rank_att,
            file_att,
            diag_att,
            pawn_att_white,
            pawn_att_black,
            rank_m,
            file_m,
            diag_m,
        ) in SQUARE_CONTROL_FOR_BOTH_ITERATOR:
            attackers_both = _attackers_mask_light_for_both(
                kings,
                knights,
                oc,
                q_and_r,
                q_and_b,
                king_att,
                knight_att,
                rank_att,
                file_att,
                diag_att,
                rank_m,
                file_m,
                diag_m,
            )
            attackers_white = (attackers_both | (pawn_att_white & pawns)) & oc_co_white
            attackers_black = (attackers_both | (pawn_att_black & pawns)) & oc_co_black
            arr[square] = get_bit_count(attackers_white) - get_bit_count(
                attackers_black
            )
        return arr

    @staticmethod
    def _attackers_mask_light_for_both(
        kings: Bitboard,
        knights: Bitboard,
        occupied: Bitboard,
        q_and_r: Bitboard,
        q_and_b: Bitboard,
        king_att: Bitboard,
        knight_att: Bitboard,
        rank_att: Dict[Bitboard, Bitboard],
        file_att: Dict[Bitboard, Bitboard],
        diag_att: Dict[Bitboard, Bitboard],
        rank_mask: Bitboard,
        file_mask: Bitboard,
        diag_mask: Bitboard,
    ) -> Bitboard:
        return (
            (king_att & kings)
            | (knight_att & knights)
            | (rank_att[rank_mask & occupied] & q_and_r)
            | (file_att[file_mask & occupied] & q_and_r)
            | (diag_att[diag_mask & occupied] & q_and_b)
        )

    def get_empty_square_map(self) -> NDArray[Shape["64"], Int]:
        """
        Returns list of 64 values, each is empty or not.
        """

        non_occupied: Bitboard = self.occupied ^ ONES

        arr = empty(shape=(64,), dtype=int)
        for square in SQUARES:
            arr[square] = (non_occupied >> square) & 1
        return arr

    def get_occupied_square_value_map(
        self, color: Color
    ) -> NDArray[Shape["64"], Double]:
        """
        Returns list of 64 values, value of a piece on each square.

        :param color: counting attacks threatened by this color
        """

        pawns = self.pawns
        knights = self.knights
        bishops = self.bishops
        rooks = self.rooks
        queens = self.queens
        kings = self.kings

        get_piece_value = self.get_piece_value
        fast_piece_type_at = self.fast_piece_type_at
        oc_co = self.occupied_co[color]

        arr = empty(shape=(64,), dtype=double)
        for square, mask in SQUARE_MASK_ITERATOR:
            arr[square] = (
                0
                if not bool((oc_co >> square) & 1)  # is not occupied by this color
                else get_piece_value(
                    color,
                    fast_piece_type_at(
                        mask, pawns, knights, bishops, rooks, queens, kings
                    ),
                    square >> 3,
                )
            )
        return arr

    @staticmethod
    def fast_piece_type_at(
        mask: Bitboard,
        pawns: Bitboard,
        knights: Bitboard,
        bishops: Bitboard,
        rooks: Bitboard,
        queens: Bitboard,
        kings: Bitboard,
    ) -> Optional[PieceType]:
        """Gets the piece type at the given square."""

        if pawns & mask:
            return PAWN
        elif knights & mask:
            return KNIGHT
        elif bishops & mask:
            return BISHOP
        elif rooks & mask:
            return ROOK
        elif queens & mask:
            return QUEEN
        elif kings & mask:
            return KING
        # should never be reached
        return None

    @staticmethod
    def generate_king_proximity_map_normalized(
        king: Square,
    ) -> NDArray[Shape["64"], Double]:
        """
        Returns list of 64 values, each distance from king.
        Normalized so that:
            distance 0 => 1
            distance 7 => 0

        :param king: square where the king is
        """

        seven: double = double(7.0)  # pff

        arr = empty(shape=(64,), dtype=double)
        for square in SQUARES:
            arr[square] = double(square_distance(square, king))
        return (seven - arr) / seven

    def get_king_proximity_map_normalized(
        self, color: Color
    ) -> NDArray[Shape["64"], Double]:
        king: Bitboard = self.kings & self.occupied_co[color]
        return KING_PROXIMITY_MAPS_NORMALIZED[king]

    def get_normalized_threats_map(self, color: Color) -> List[double]:
        """
        Returns list of 64 values, accumulated threats on each square.

        Each value is normalized to between 0 and 1, with assumption of maximum threats being 8.
        In theory there can be more attackers on a square than 8, but in practice it's hard to think of
        a realistic position with more than 4-5 attackers on single square

        :param color: counting threats threatened by this color
        """

        return [
            double(min(1.0, get_bit_count(self.threats_mask(color, square)) / 8))
            for square in SQUARES
        ]

    def has_white_bishop(self, color: Color) -> bool:
        return self.bishops & self.occupied_co[color] & BB_LIGHT_SQUARES != 0

    def has_black_bishop(self, color: Color) -> bool:
        return self.bishops & self.occupied_co[color] & BB_DARK_SQUARES != 0

    def get_mobility(self, color: Color) -> int:
        mobility: int = 0
        oc_co: Bitboard = self.occupied_co[color]
        noc_co: Bitboard = ~oc_co

        for square in scan_forward(self.queens & oc_co):
            mobility += get_bit_count(self.attacks_mask(square) & noc_co)

        for square in scan_forward(self.rooks & oc_co):
            mobility += get_bit_count(self.attacks_mask(square) & noc_co)

        for square in scan_forward(self.bishops & oc_co):
            mobility += get_bit_count(self.attacks_mask(square) & noc_co)

        for square in scan_forward(self.knights & oc_co):
            mobility += get_bit_count(self.attacks_mask(square) & noc_co)

        return mobility

    def get_threats(self, color: Color) -> int:
        threats: int = 0
        oc_nco: Bitboard = self.occupied_co[not color]

        # sum of xray attacks on opposite color king/queen
        for square in scan_forward(self.queens & oc_nco):
            threats += 9 * get_bit_count(self.threats_mask(color, square))

        for square in scan_forward(self.rooks & oc_nco):
            threats += 5 * get_bit_count(self.threats_mask(color, square))

        for square in scan_forward((self.bishops | self.knights) & oc_nco):
            threats += 3 * get_bit_count(self.threats_mask(color, square))

        return threats

    def get_king_threats(self, color: Color) -> int:
        return get_bit_count(
            self.threats_mask(
                color, self.mask_to_square(self.kings & self.occupied_co[not color])
            )
        )

    # def get_checkmate(self) -> bool:
    #     """
    #     Identifying checkmate in an optimized way.
    #
    #     Is already assumed that king is in check.
    #     """

    def get_direct_threats(self, color: Color) -> int:
        threats: int = 0
        oc_nco: Bitboard = self.occupied_co[not color]

        for square in scan_forward(self.queens & oc_nco):
            threats += 9 * get_bit_count(self.direct_threats_mask(color, square))

        for square in scan_forward(self.rooks & oc_nco):
            threats += 5 * get_bit_count(self.direct_threats_mask(color, square))

        for square in scan_forward((self.bishops | self.knights) & oc_nco):
            threats += 3 * get_bit_count(self.direct_threats_mask(color, square))

        return threats

    def pawns_on_light_squares(self, color: Color) -> int:
        return get_bit_count(self.occupied_co[color] & BB_LIGHT_SQUARES & self.pawns)

    def pieces_on_light_squares(self, color: Color) -> int:
        return get_bit_count(self.occupied_co[color] & BB_LIGHT_SQUARES & ~self.pawns)

    def pawns_on_dark_squares(self, color: Color) -> int:
        return get_bit_count(self.occupied_co[color] & BB_DARK_SQUARES & self.pawns)

    def pieces_on_dark_squares(self, color: Color) -> int:
        return get_bit_count(self.occupied_co[color] & BB_DARK_SQUARES & ~self.pawns)

    def get_moving_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.from_square) or 0

    def get_captured_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    def simple_outcome(self) -> Optional[Outcome]:
        """"""

        if self.is_checkmate():
            return Outcome(Termination.CHECKMATE, not self.turn)
        if self.is_insufficient_material():
            return Outcome(Termination.INSUFFICIENT_MATERIAL, None)

        if self.can_claim_ten_moves():
            return Outcome(Termination.FIFTY_MOVES, None)
        if self.simple_can_claim_threefold_repetition():
            return Outcome(Termination.THREEFOLD_REPETITION, None)

        return None

    def can_claim_ten_moves(self) -> bool:
        return self._is_halfmoves(20)

    def simple_can_claim_threefold_repetition(self) -> bool:
        """"""

        transposition_key = self._transposition_key()
        transpositions: Counter[Hashable] = collections.Counter()
        transpositions.update((transposition_key,))

        # Count positions.
        switchyard = []
        while self.move_stack:
            # move = self.pop()
            move = self.light_pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(),))

        while switchyard:
            # self.push(switchyard.pop())
            self.light_push(switchyard.pop())

        # Threefold repetition occured.
        if (
            transpositions[transposition_key] >= 2
        ):  # changed to 2 to avoid engine repeat positions
            return True

        return False

    def light_push(self: BoardT, move: Move) -> None:
        """
        Updates the position with the given *move* and puts it onto the move stack.
        """

        # Push move and remember board state.
        move = self._to_chess960(move)
        board_state = self._board_state()
        self.move_stack.append(
            self._from_chess960(
                self.chess960,
                move.from_square,
                move.to_square,
                move.promotion,
                move.drop,
            )
        )
        self._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        from_bb = BB_SQUARES[move.from_square]
        to_bb = BB_SQUARES[move.to_square]

        promoted = bool(self.promoted & from_bb)
        piece_type = cast(int, self._remove_piece_at(move.from_square))
        capture_square = move.to_square
        captured_piece_type = self.piece_type_at(capture_square)

        # Handle special pawn moves.
        if piece_type == PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and square_rank(move.from_square) == 1:
                self.ep_square = move.from_square + 8
            elif diff == -16 and square_rank(move.from_square) == 6:
                self.ep_square = move.from_square - 8
            elif (
                move.to_square == ep_square
                and abs(diff) in [7, 9]
                and not captured_piece_type
            ):
                # Remove pawns captured en passant.
                down = -8 if self.turn == True else 8
                capture_square = ep_square + down
                captured_piece_type = self._remove_piece_at(capture_square)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = piece_type == KING and self.occupied_co[self.turn] & to_bb
        if castling:
            a_side = square_file(move.to_square) < square_file(move.from_square)

            self._remove_piece_at(move.from_square)
            self._remove_piece_at(move.to_square)

            if a_side:
                self._set_piece_at(C1 if self.turn == True else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn == True else D8, ROOK, self.turn)
            else:
                self._set_piece_at(G1 if self.turn == True else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn == True else F8, ROOK, self.turn)

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            self._set_piece_at(move.to_square, piece_type, self.turn, promoted)

            if captured_piece_type:
                self._push_capture(
                    move, capture_square, captured_piece_type, was_promoted
                )

        # Swap turn.
        self.turn = not self.turn

    def get_board_state(self) -> Dict[str, Bitboard]:
        return {
            "pawns": self.pawns,
            "knights": self.knights,
            "bishops": self.bishops,
            "rooks": self.rooks,
            "queens": self.queens,
            "kings": self.kings,
            "occupied_w": self.occupied_co[True],
            "occupied_b": self.occupied_co[False],
            "occupied": self.occupied,
            "castling_rights": self.castling_rights,
        }

    def lighter_push(self, move: Move) -> Dict[str, Bitboard]:
        """
        Updates the position with the given *move* and puts it onto the move stack.
        """

        board_state = self.get_board_state()

        # Push move and remember board state.
        move = self._to_chess960(move)

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        from_bb = BB_SQUARES[move.from_square]
        to_bb = BB_SQUARES[move.to_square]

        promoted = bool(self.promoted & from_bb)
        piece_type = cast(int, self._remove_piece_at(move.from_square))
        capture_square = move.to_square
        captured_piece_type = self.piece_type_at(capture_square)

        # Handle special pawn moves.
        if piece_type == PAWN:
            diff = move.to_square - move.from_square

            if diff == 16 and square_rank(move.from_square) == 1:
                self.ep_square = move.from_square + 8
            elif diff == -16 and square_rank(move.from_square) == 6:
                self.ep_square = move.from_square - 8
            elif (
                move.to_square == ep_square
                and abs(diff) in [7, 9]
                and not captured_piece_type
            ):
                # Remove pawns captured en passant.
                down = -8 if self.turn == True else 8
                capture_square = ep_square + down
                captured_piece_type = self._remove_piece_at(capture_square)

        # Promotion.
        if move.promotion:
            promoted = True
            piece_type = move.promotion

        # Castling.
        castling = piece_type == KING and self.occupied_co[self.turn] & to_bb
        if castling:
            a_side = square_file(move.to_square) < square_file(move.from_square)

            self._remove_piece_at(move.from_square)
            self._remove_piece_at(move.to_square)

            if a_side:
                self._set_piece_at(C1 if self.turn == True else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn == True else D8, ROOK, self.turn)
            else:
                self._set_piece_at(G1 if self.turn == True else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn == True else F8, ROOK, self.turn)

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            self._set_piece_at(move.to_square, piece_type, self.turn, promoted)

            if captured_piece_type:
                self._push_capture(
                    move, capture_square, captured_piece_type, was_promoted
                )

        # Swap turn.
        self.turn = not self.turn

        return board_state

    def light_pop(self) -> Move:
        """
        Restores the previous position without
        """

        move = self.move_stack.pop()
        self.light_restore(self._stack.pop())
        self.turn = not self.turn
        return move

    def restore_board_state(self, state: Dict):
        self.pawns = state["pawns"]
        self.knights = state["knights"]
        self.bishops = state["bishops"]
        self.rooks = state["rooks"]
        self.queens = state["queens"]
        self.kings = state["kings"]
        self.occupied_co[True] = state["occupied_w"]
        self.occupied_co[False] = state["occupied_b"]
        self.occupied = state["occupied"]
        self.castling_rights = state["castling_rights"]

    def lighter_pop(self, state: Dict) -> None:
        self.turn = not self.turn
        self.restore_board_state(state)

    def light_restore(self, board_state: _BoardState) -> None:
        """
        Restores only the positions of the pieces/pawns.
        """

        self.pawns = board_state.pawns
        self.knights = board_state.knights
        self.bishops = board_state.bishops
        self.rooks = board_state.rooks
        self.queens = board_state.queens
        self.kings = board_state.kings

        self.occupied_co[True] = board_state.occupied_w
        self.occupied_co[False] = board_state.occupied_b
        self.occupied = board_state.occupied

    def generate_pseudo_legal_moves(self, from_mask: Bitboard = BB_ALL, to_mask: Bitboard = BB_ALL) -> Iterator[Move]:
        """
        Override from original changing the order to start generating from pawn captures, then other captures.
        """

        our_pieces = self.occupied_co[self.turn]
        opp_pieces = self.occupied_co[not self.turn]

        pawns = self.pawns & self.occupied_co[self.turn] & from_mask
        if pawns:
            # Generate pawn captures.
            for from_square in scan_reversed(pawns):
                targets = (
                    BB_PAWN_ATTACKS[self.turn][from_square] &
                    self.occupied_co[not self.turn] & to_mask)

                for to_square in scan_reversed(targets):
                    if square_rank(to_square) in [0, 7]:
                        yield Move(from_square, to_square, QUEEN)
                        yield Move(from_square, to_square, ROOK)
                        yield Move(from_square, to_square, BISHOP)
                        yield Move(from_square, to_square, KNIGHT)
                    else:
                        yield Move(from_square, to_square)

        # Generate en passant captures.
        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

        # Generate piece moves.
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            attacks = self.attacks_mask(from_square) & to_mask
            captures = attacks & opp_pieces
            moves = attacks & ~opp_pieces & ~our_pieces

            for to_square in scan_reversed(captures):
                yield Move(from_square, to_square)

            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        # Generate castling moves.
        if from_mask & self.kings:
            yield from self.generate_castling_moves(from_mask, to_mask)

        # Prepare pawn advance generation.
        if self.turn:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if not self.turn else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if not self.turn else -16)
            yield Move(from_square, to_square)


KING_PROXIMITY_MAPS_NORMALIZED: Dict[Bitboard, NDArray[Shape["64"], Double]] = {
    mask: Board.generate_king_proximity_map_normalized(square) for mask, square in zip(BB_SQUARES, SQUARES)
}
pawn_attacks_white: List[Bitboard] = BB_PAWN_ATTACKS[False]
pawn_attacks_black: List[Bitboard] = BB_PAWN_ATTACKS[True]
SQUARE_MASK_ITERATOR: List[Tuple[int, Bitboard]] = list(zip(SQUARES, BB_SQUARES))
SQUARE_CONTROL_FOR_BOTH_ITERATOR: List[
    Tuple[
        Bitboard,
        Bitboard,
        Bitboard,
        Dict[Bitboard, Bitboard],
        Dict[Bitboard, Bitboard],
        Dict[Bitboard, Bitboard],
        Bitboard,
        Bitboard,
        Bitboard,
        Bitboard,
        Bitboard,
    ]
] = cast(
    List[
        Tuple[
            Bitboard,
            Bitboard,
            Bitboard,
            Dict[Bitboard, Bitboard],
            Dict[Bitboard, Bitboard],
            Dict[Bitboard, Bitboard],
            Bitboard,
            Bitboard,
            Bitboard,
            Bitboard,
            Bitboard,
        ]
    ],
    list(
        zip(
            SQUARES,
            BB_KING_ATTACKS,
            BB_KNIGHT_ATTACKS,
            BB_RANK_ATTACKS,
            BB_FILE_ATTACKS,
            BB_DIAG_ATTACKS,
            BB_PAWN_ATTACKS[False],  # false for white
            BB_PAWN_ATTACKS[True],  # true for black
            BB_RANK_MASKS,
            BB_FILE_MASKS,
            BB_DIAG_MASKS,
        )
    ),
)
PAWN_VALUES: Dict[int, double] = {
    rank: double(1.0177) ** (max(0, rank - 2) ** 3) for rank in range(7)
}
KING_ATTACKS: Dict[Bitboard, Bitboard] = {
    mask: BB_KING_ATTACKS[square] for square, mask in zip(SQUARES, BB_SQUARES)
}
ONES: Bitboard = (2**64 - 1)
