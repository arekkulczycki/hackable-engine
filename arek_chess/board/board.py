# -*- coding: utf-8 -*-
"""
Handling the chessboard and calculating features of a position for RL training observation.
"""
import collections
from typing import List, Optional, Counter, Hashable

from chess import (
    Board as ChessBoard,
    Bitboard,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    scan_forward,
    square_rank,
    Square,
    BB_SQUARES,
    Color,
    BB_KING_ATTACKS,
    BB_KNIGHT_ATTACKS,
    BB_RANK_ATTACKS,
    BB_FILE_ATTACKS,
    BB_DIAG_ATTACKS,
    BB_PAWN_ATTACKS,
    SQUARES,
    BB_LIGHT_SQUARES,
    BB_DARK_SQUARES,
    PieceType,
    Move,
    Outcome,
    Termination,
)
from numpy import double


class Board(ChessBoard):
    """
    Handling the chessboard and calculating features of a position.
    """

    @staticmethod
    def get_bit_count(bb: Bitboard) -> int:
        return bin(bb).count("1")

    def pieces_mask_both_no_kings(self, piece_type: PieceType) -> Bitboard:
        if piece_type == PAWN:
            bb = self.pawns
        elif piece_type == KNIGHT:
            bb = self.knights
        elif piece_type == BISHOP:
            bb = self.bishops
        elif piece_type == ROOK:
            bb = self.rooks
        elif piece_type == QUEEN:
            bb = self.queens
        else:
            assert False, f"expected PieceType, got {piece_type!r}"

        return bb

    def get_material_simple(self, color: bool) -> int:
        return (
            self.get_bit_count(self.pieces_mask(piece_type=PAWN, color=color))
            + 3 * self.get_bit_count(self.pieces_mask(piece_type=KNIGHT, color=color))
            + 3 * self.get_bit_count(self.pieces_mask(piece_type=BISHOP, color=color))
            + 5 * self.get_bit_count(self.pieces_mask(piece_type=ROOK, color=color))
            + 9 * self.get_bit_count(self.pieces_mask(piece_type=QUEEN, color=color))
        )

    def get_material_simple_both(self) -> int:
        return (
            self.get_bit_count(self.pawns)
            + 3 * self.get_bit_count(self.knights)
            + 3 * self.get_bit_count(self.bishops)
            + 5 * self.get_bit_count(self.rooks)
            + 9 * self.get_bit_count(self.queens)
        )

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

        return double(1.0177) ** (
            max(0, rank - 2) ** 3
        )  # equals 9 (queen) on the last rank, equals 1 on low ranks

    def get_material_no_pawns(self, color: bool) -> int:
        oc_co: Bitboard = self.occupied_co[color]

        return (
            3 * self.get_bit_count(self.knights & oc_co)
            + 3 * self.get_bit_count(self.bishops & oc_co)
            + 5 * self.get_bit_count(self.rooks & oc_co)
            + 9 * self.get_bit_count(self.queens & oc_co)
        )

    def get_material_no_pawns_both(self) -> int:
        return (
            3 * self.get_bit_count(self.knights)
            + 3 * self.get_bit_count(self.bishops)
            + 5 * self.get_bit_count(self.rooks)
            + 9 * self.get_bit_count(self.queens)
        )

    def mask_to_square(self, mask: Bitboard) -> Square:
        return BB_SQUARES.index(mask)

    def mask_to_squares(self, mask: Bitboard) -> List[Square]:
        squares = []
        for square in scan_forward(mask):
            squares.append(square)
        return squares

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

        attackers = (BB_KNIGHT_ATTACKS[square] & self.knights) | (BB_PAWN_ATTACKS[not color][square] & self.pawns)

        return attackers & self.occupied_co[color]

    def get_king_mobility(self, color: Color) -> int:
        """Own pieces/pawns and enemy pawns are considered shield."""
        shield = self.occupied_co[color] | (self.occupied_co[False] & self.pawns)

        return self.get_bit_count(
            BB_KING_ATTACKS[self.mask_to_square(self.kings & self.occupied_co[color])] & ~shield
        )

    def get_normalized_threats_map(self, color: Color) -> List[double]:
        """
        Returns list of 64 values, accumulated threats on each square.

        Each value is normalized to between 0 and 1, with assumption of maximum threats being 8.
        In theory there can be more attackers on a square than 8, but in practice it's hard to think of
        a realistic position with more than 4-5 attackers on single square

        :param color: counting attacks threatened by this color
        """

        return [
            double(min(1.0, self.get_bit_count(self.threats_mask(color, square)) / 8))
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
            mobility += self.get_bit_count(
                self.attacks_mask(square) & noc_co
            )

        for square in scan_forward(self.rooks & oc_co):
            mobility += self.get_bit_count(
                self.attacks_mask(square) & noc_co
            )

        for square in scan_forward(self.bishops & oc_co):
            mobility += self.get_bit_count(
                self.attacks_mask(square) & noc_co
            )

        for square in scan_forward(self.knights & oc_co):
            mobility += self.get_bit_count(
                self.attacks_mask(square) & noc_co
            )

        return mobility

    def get_threats(self, color: Color) -> int:
        threats: int = 0
        oc_nco: Bitboard = self.occupied_co[not color]

        # sum of xray attacks on opposite color king/queen
        for square in scan_forward(self.queens & oc_nco):
            threats += 9 * self.get_bit_count(self.threats_mask(color, square))

        for square in scan_forward(self.rooks & oc_nco):
            threats += 5 * self.get_bit_count(self.threats_mask(color, square))

        for square in scan_forward(
            (self.bishops | self.knights) & oc_nco
        ):
            threats += 3 * self.get_bit_count(self.threats_mask(color, square))

        return threats

    def get_king_threats(self, color: Color) -> int:
        return self.get_bit_count(
            self.threats_mask(
                color, self.mask_to_square(self.kings & self.occupied_co[not color])
            )
        )

    def get_direct_threats(self, color: Color) -> int:
        threats: int = 0
        oc_nco: Bitboard = self.occupied_co[not color]

        for square in scan_forward(self.queens & oc_nco):
            threats += 9 * self.get_bit_count(self.direct_threats_mask(color, square))

        for square in scan_forward(self.rooks & oc_nco):
            threats += 5 * self.get_bit_count(self.direct_threats_mask(color, square))

        for square in scan_forward(
            (self.bishops | self.knights) & oc_nco
        ):
            threats += 3 * self.get_bit_count(self.direct_threats_mask(color, square))

        return threats

    def pawns_on_light_squares(self, color: Color) -> int:
        return self.get_bit_count(
            self.occupied_co[color] & BB_LIGHT_SQUARES & self.pawns
        )

    def pieces_on_light_squares(self, color: Color) -> int:
        return self.get_bit_count(
            self.occupied_co[color] & BB_LIGHT_SQUARES & ~self.pawns
        )

    def pawns_on_dark_squares(self, color: Color) -> int:
        return self.get_bit_count(
            self.occupied_co[color] & BB_DARK_SQUARES & self.pawns
        )

    def pieces_on_dark_squares(self, color: Color) -> int:
        return self.get_bit_count(
            self.occupied_co[color] & BB_DARK_SQUARES & ~self.pawns
        )

    def get_moving_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.from_square) or 0

    def get_captured_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    def simple_outcome(self) -> Optional[Outcome]:
        """"""

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
            move = self.pop()
            switchyard.append(move)

            if self.is_irreversible(move):
                break

            transpositions.update((self._transposition_key(),))

        while switchyard:
            self.push(switchyard.pop())

        # Threefold repetition occured.
        if (
            transpositions[transposition_key] >= 2
        ):  # changed to 2 to avoid engine repeat positions
            return True

        return False
