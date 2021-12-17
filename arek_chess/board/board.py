# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from typing import Tuple, Dict, Iterator, List, Generator

import numpy
from chess import (
    Board as ChessBoard,
    scan_reversed,
    Move,
    square_rank,
    _attack_table,
    Square,
    Bitboard,
    BB_KNIGHT_ATTACKS,
    BB_KING_ATTACKS,
    WHITE,
    BLACK,
    KING,
    square_file,
    ROOK,
    F8,
    G8,
    D8,
    C8,
    C1,
    D1,
    G1,
    F1,
    BB_SQUARES,
    QUEEN,
    BISHOP,
    PAWN,
    KNIGHT,
    shift_2_down,
    shift_up,
    shift_2_up,
    shift_down,
    BB_ALL,
    BB_RANK_6,
    BB_RANK_5,
    BB_RANK_4,
    BB_RANK_3,
    BB_PAWN_ATTACKS,
    BB_RANK_1,
    BB_RANK_8,
)

BB_DIAG_MASKS: List[int]
BB_DIAG_ATTACKS: List[Dict[int, int]]
BB_FILE_MASKS: List[int]
BB_FILE_ATTACKS: List[Dict[int, int]]
BB_RANK_MASKS: List[int]
BB_RANK_ATTACKS: List[Dict[int, int]]

BB_DIAG_MASKS, BB_DIAG_ATTACKS = _attack_table([-9, -7, 7, 9])
BB_FILE_MASKS, BB_FILE_ATTACKS = _attack_table([-8, 8])
BB_RANK_MASKS, BB_RANK_ATTACKS = _attack_table([-1, 1])
FLAT_BB_PAWN_ATTACKS = [item for sublist in BB_PAWN_ATTACKS for item in sublist]


def lsb(bb: Bitboard) -> int:
    return (bb & -bb).bit_length() - 1


class Board(ChessBoard):
    """
    Class_docstring
    """

    @staticmethod
    def calculate_score(action: List[float], params: List[float], piece_type=None):
        # TODO: when no pieces on board, transform action params of pieces to columns of pawns to use! train pawn endgames separately
        # piece_type_bonus = (
        #     0  # 10.0 * action[piece_type - 1 + 4] if piece_type is not None else 0
        # )
        return numpy.dot(action, params)  # + piece_type_bonus

    @staticmethod
    def get_bit_count(bb: Bitboard) -> int:
        return bin(bb).count("1")

    def get_board_state(self) -> Dict:
        return {
            "pawns": self.pawns,
            "knights": self.knights,
            "bishops": self.bishops,
            "rooks": self.rooks,
            "queens": self.queens,
            "kings": self.kings,
            "occupied_w": self.occupied_co[WHITE],
            "occupied_b": self.occupied_co[BLACK],
            "occupied": self.occupied,
            "promoted": self.promoted,
            "turn": self.turn,
            "castling_rights": self.castling_rights,
            "ep_square": self.ep_square,
        }

    def restore_board_state(self, state: Dict):
        self.pawns = state["pawns"]
        self.knights = state["knights"]
        self.bishops = state["bishops"]
        self.rooks = state["rooks"]
        self.queens = state["queens"]
        self.kings = state["kings"]
        self.occupied_co[WHITE] = state["occupied_w"]
        self.occupied_co[BLACK] = state["occupied_b"]
        self.occupied = state["occupied"]
        self.promoted = state["promoted"]
        self.turn = state["turn"]
        self.castling_rights = state["castling_rights"]
        self.ep_square = state["ep_square"]

    def push_no_stack(self, move: Move) -> Dict:
        # move = self._to_chess960(move)
        board_state = self.get_board_state()
        self.castling_rights = self.clean_castling_rights()  # Before pushing stack
        # self.move_stack.append(self._from_chess960(self.chess960, move.from_square, move.to_square, move.promotion, move.drop))
        # self._stack.append(board_state)

        # Reset en passant square.
        ep_square = self.ep_square
        self.ep_square = None

        # Increment move counters.
        self.halfmove_clock += 1
        if self.turn == BLACK:
            self.fullmove_number += 1

        # Zero the half-move clock.
        if self.is_zeroing(move):
            self.halfmove_clock = 0

        from_bb = BB_SQUARES[move.from_square]
        to_bb = BB_SQUARES[move.to_square]

        promoted = bool(self.promoted & from_bb)
        piece_type = self._remove_piece_at(move.from_square)
        assert (
            piece_type is not None
        ), f"push() expects move to be pseudo-legal, but got {move} in {self.board_fen()}"
        capture_square = move.to_square
        captured_piece_type = self.piece_type_at(capture_square)

        # Update castling rights.
        self.castling_rights &= ~to_bb & ~from_bb
        if piece_type == KING and not promoted:
            if self.turn == WHITE:
                self.castling_rights &= ~BB_RANK_1
            else:
                self.castling_rights &= ~BB_RANK_8
        elif captured_piece_type == KING and not self.promoted & to_bb:
            if self.turn == WHITE and square_rank(move.to_square) == 7:
                self.castling_rights &= ~BB_RANK_8
            elif self.turn == BLACK and square_rank(move.to_square) == 0:
                self.castling_rights &= ~BB_RANK_1

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
                down = -8 if self.turn == WHITE else 8
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
                self._set_piece_at(C1 if self.turn == WHITE else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn == WHITE else D8, ROOK, self.turn)
            else:
                self._set_piece_at(G1 if self.turn == WHITE else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn == WHITE else F8, ROOK, self.turn)

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

    def light_push(self, move: Move, state_required: bool = True) -> Dict:
        board_state = self.get_board_state() if state_required else {}

        to_bb = BB_SQUARES[move.to_square]

        piece_type = self._remove_piece_at(move.from_square)

        promoted = False
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
                self._set_piece_at(C1 if self.turn == WHITE else C8, KING, self.turn)
                self._set_piece_at(D1 if self.turn == WHITE else D8, ROOK, self.turn)
            else:
                self._set_piece_at(G1 if self.turn == WHITE else G8, KING, self.turn)
                self._set_piece_at(F1 if self.turn == WHITE else F8, ROOK, self.turn)

        # Put the piece on the target square.
        if not castling:
            was_promoted = bool(self.promoted & to_bb)
            if piece_type:
                self._set_piece_at(move.to_square, piece_type, self.turn, promoted)

            captured_piece_type = self.piece_type_at(move.to_square)
            if captured_piece_type:
                self._push_capture(
                    move, move.to_square, captured_piece_type, was_promoted
                )

        # Swap turn.
        self.turn = not self.turn

        return board_state

    def light_pop(self, state: Dict):
        self.restore_board_state(state)

    #     # perform only necessary part or restore
    #     self.pawns = self.pawns
    #     self.knights = self.knights
    #     self.bishops = self.bishops
    #     self.rooks = self.rooks
    #     self.queens = self.queens
    #     self.kings = self.kings
    #
    #     self.occupied_co[WHITE] = self.occupied_w
    #     self.occupied_co[BLACK] = self.occupied_b
    #     self.occupied = self.occupied
    #
    #     self.promoted = self.promoted
    #
    #     self.turn = self.turn
    #     self.castling_rights = self.castling_rights
    #     self.ep_square = self.ep_square
    #     self.halfmove_clock = self.halfmove_clock
    #     self.fullmove_number = self.fullmove_number

    def get_material_delta(self, captured_piece_type: int):
        # if white to move then capture is + in material, otherwise is -
        sign = 1 if self.turn else -1

        # color of the captured piece is opposite to turn (side making the move)
        return self.get_piece_value(captured_piece_type, not self.turn) * sign

    def value_of_defended_pieces(
        self, color: bool, square: Square, masks: List[Bitboard]
    ):
        """
        Calculate sum value of all pieces being attacked from the given square
        :param color: the color of pieces being attacked
        """

        bb_square = BB_SQUARES[square]
        return self.defended_accumulated_value(color, bb_square, masks)

    def value_of_defended_pieces_by(
        self, color: bool, attackers_mask: Bitboard, masks: List[Bitboard]
    ):
        """
        Calculate sum of value of all pieces being defended by defenders which are on attackers_mask squares
        """

        under_attack = 0.0
        for bb_square in self.scan_bb_forward(attackers_mask):
            under_attack += self.attacked_accumulated_value(not color, bb_square, masks)
        return under_attack

    def get_safety_delta(
        self, color: bool, move: Move, moved_piece_type: int, captured_piece_type: int
    ):
        """

        :param color: color of the side to calculate safety for
        :param move:
        :param moved_piece_type:
        :param captured_piece_type:
        :return:
        """

        masks_before = [self.pawns, self.knights, self.bishops, self.rooks, self.queens]

        # if move color was the same as calculated color (self.turn == color) then capture doesn't matter
        if self.turn == color:
            # TODO: potentially take len below without initializing SquareSet objects?

            # safety coming from protecting other squares by the moved piece, before the move
            safety_before = self.value_of_defended_pieces(
                color,
                move.from_square,
                masks_before,
            )

            # safety coming from the moved piece being protected, before
            if moved_piece_type != KING:
                safety_before += self.num_attackers(
                    color, move.from_square
                ) * (self.get_piece_value(PAWN, color, square=move.to_square) if moved_piece_type == PAWN else 3)

            # self.push(move)
            board_state = self.light_push(move)

            masks_after = [
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
            ]

            # safety coming from protecting other squares by the moved piece, after
            safety_after = self.value_of_defended_pieces(
                color,
                move.to_square,
                masks_after,
            )

            # safety coming from the moved piece being protected, after
            if moved_piece_type != KING:
                safety_after += self.num_attackers(
                    color, move.to_square
                ) * (1 if moved_piece_type == PAWN else 3)

            # self.pop()
            self.light_pop(board_state)

            return safety_after - safety_before

        else:  # moving side is opposite to calculated safety
            # in this case the safety change is:
            #  - if captured, equal to all attacks on the captured piece

            if captured_piece_type:
                safety_loss = self.num_attackers(
                    color, move.to_square
                ) * (self.get_piece_value(PAWN, color, square=move.to_square) if moved_piece_type == PAWN else 3)

                safety_loss += self.value_of_defended_pieces(
                    color, move.to_square, masks_before
                )

                return -safety_loss

            #  - otherwise, if piece got in the way of protectors, for all the attackers on the current piece
            #    must be found if they protected a piece in previous position

            safety_loss = 0.0

            # attackers_before = self.attackers(color, move.to_square)
            attackers_mask_before = self.attackers_mask(color, move.to_square)

            # self.push(move)
            board_state = self.light_push(move)

            masks_after = [
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
            ]

            # attackers_after = self.attackers(color, move.to_square)
            attackers_mask_after = self.attackers_mask(color, move.to_square)
            safety_loss -= self.value_of_defended_pieces_by(
                color, attackers_mask_before, masks_after
            )

            # self.pop()
            self.light_pop(board_state)

            safety_loss += self.value_of_defended_pieces_by(
                color, attackers_mask_after, masks_before
            )

            return -safety_loss

    def defended_accumulated_value(
        self, color: bool, bb_square: Bitboard, masks: List[Bitboard]
    ):
        """
        :param color: the color of pieces being defended
        :param bb_square: square from which attacks are threatened
        """

        attacked_mask = self.defended_mask(color, bb_square)
        v = 0.0
        for mask, factor in zip(masks, [1, 3, 3, 3, 3]):
            both = attacked_mask & mask
            if both:
                v += self.get_bit_count(both) * factor
        return v

    def attacked_accumulated_value(
        self, color: bool, bb_square: Bitboard, masks: List[Bitboard]
    ):
        """
        :param color: the color of pieces being attacked
        :param bb_square: square from which attacks are threatened
        """

        attacked_mask = self.attacked_mask(color, bb_square)
        v = 0.0
        for mask, factor in zip(masks, [1, 3, 3, 4.5, 9]):
            both = attacked_mask & mask
            if both:
                v += self.get_bit_count(both) * factor
        return v

    def value_of_attacked_pieces(
        self, color: bool, attackers_mask: Bitboard, masks: List[Bitboard]
    ):
        """
        Calculate sum of value of all pieces being attacked by attackers which are on attackers_mask squares
        :param color: the color of pieces being attacked
        """

        under_attack = 0.0
        for bb_square in self.scan_bb_forward(attackers_mask):
            under_attack += self.attacked_accumulated_value(color, bb_square, masks)
        return under_attack

    def get_under_attack_delta(
        self, color: bool, move: Move, moved_piece_type: int, captured_piece: int
    ):
        """

        :param color: color of the side to calculate under_attack for
        :param move:
        :param moved_piece_type:
        :param captured_piece:
        :return:
        """

        masks_before = [self.pawns, self.knights, self.bishops, self.rooks, self.queens]

        # if calculating same color as making the move
        if color == self.turn:
            # attacks on the moved piece are included in the accumulated values for all attackers

            # all the pieces that attacked the square from where the move was made (maybe opened a discovered attack)
            attackers_mask_before = self.attackers_mask(not self.turn, move.from_square)

            # self.push(move)
            board_state = self.light_push(move)

            masks_after = [
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
            ]

            # all the pieces that attack the square to where the move was made (maybe shielded other attacks)
            attackers_mask_after = self.attackers_mask(self.turn, move.to_square)

            # self.pop()
            self.light_pop(board_state)

            # find sum of attacks value, of all above attackers, before the move
            under_attack_before = 0.0
            under_attack_before += self.value_of_attacked_pieces(
                self.turn, attackers_mask_before, masks_before
            )
            under_attack_before += self.value_of_attacked_pieces(
                self.turn, attackers_mask_after, masks_before
            )

            # self.push(move)
            self.light_push(move, False)

            # find sum of attacks value after the move
            under_attack_after = 0.0
            under_attack_after += self.value_of_attacked_pieces(
                not self.turn, attackers_mask_before, masks_after
            )
            under_attack_after += self.value_of_attacked_pieces(
                not self.turn, attackers_mask_after, masks_after
            )

            # self.pop()
            self.light_pop(board_state)

            return under_attack_after - under_attack_before

        # calculating for opposition side
        else:
            attacked_before = self.value_of_attacked_pieces(
                not self.turn, BB_SQUARES[move.from_square], masks_before
            )

            # unblocked_attackers = self.attackers(not self.turn, move.from_square)
            unblocked_attackers_mask = self.attackers_mask(
                not self.turn, move.from_square
            )

            # self.push(move)
            board_state = self.light_push(move)

            masks_after = [
                self.pawns,
                self.knights,
                self.bishops,
                self.rooks,
                self.queens,
            ]

            # blocked_attackers = self.attackers(not self.turn, move.to_square)
            blocked_attackers_mask = self.attackers_mask(not self.turn, move.to_square)

            # minus all the blocked attacks from previous position
            blocked_attacked_after = 0.0
            for attacker_bb in self.scan_bb_forward(
                (blocked_attackers_mask & self.queens)
                | (blocked_attackers_mask & self.rooks)
                | (blocked_attackers_mask & self.bishops)
            ):
                blocked_attacked_after += self.value_of_attacked_pieces(
                    self.turn, attacker_bb, masks_after
                )

            # plus all unblocked attacks
            # unblocked_attacked_after = 0.0
            for attacker_bb in self.scan_bb_forward(
                (unblocked_attackers_mask & self.queens)
                | (unblocked_attackers_mask & self.rooks)
                | (unblocked_attackers_mask & self.bishops)
            ):
                blocked_attacked_after -= self.value_of_attacked_pieces(
                    self.turn, attacker_bb, masks_after
                )

            attacked_after = self.value_of_attacked_pieces(
                self.turn, BB_SQUARES[move.to_square], masks_after
            )

            # self.pop()
            self.light_pop(board_state)

            blocked_attacked_before = 0.0
            for attacker_bb in self.scan_bb_forward(
                (blocked_attackers_mask & self.queens)
                | (blocked_attackers_mask & self.rooks)
                | (blocked_attackers_mask & self.bishops)
            ):
                # blocked_attacked_before += get_under_attack(attacker, not self.turn)
                blocked_attacked_before += self.value_of_attacked_pieces(
                    not self.turn, attacker_bb, masks_before
                )

            return (
                attacked_after
                - attacked_before
                + blocked_attacked_after
                - blocked_attacked_before
            )

    def get_material_and_safety(self, color: bool) -> Tuple[float, float, float]:
        material = 0.0
        safety = 0.0
        under_attack = 0.0
        n_pawns = 0

        for pawn in self.pieces(piece_type=PAWN, color=color):
            n_pawns += 1
            rank = square_rank(pawn) if color else 7 - square_rank(pawn)
            val = self.get_piece_value(PAWN, color, rank=rank)
            material += val

            safety += self.num_attackers(color, pawn) * val
            under_attack += self.num_attackers(not color, pawn) * val

        for knight in self.pieces(piece_type=KNIGHT, color=color):
            val = 3
            material += val

            safety += self.num_attackers(color, knight) * val
            under_attack += self.num_attackers(not color, knight) * val

        bishop_pair = 0.0
        for bishop in self.pieces(piece_type=BISHOP, color=color):
            val = self.get_piece_value(
                BISHOP, color, n_pawns=n_pawns, bishop_pair=bishop_pair
            )
            material += val

            safety += self.num_attackers(color, bishop) * val
            under_attack += self.num_attackers(not color, bishop) * val

            bishop_pair += 1.0

        for rook in self.pieces(piece_type=ROOK, color=color):
            val = 4.5
            material += val

            safety += self.num_attackers(color, rook) * val
            under_attack += self.num_attackers(not color, rook) * val

        for queen in self.pieces(piece_type=QUEEN, color=color):
            val = 9
            material += val

            safety += self.num_attackers(color, queen) * val
            under_attack += self.num_attackers(not color, queen) * val

        return material, safety, under_attack

    def mobility_mask(self, color: bool, bb_square: Bitboard) -> Bitboard:
        # bb_square = BB_SQUARES[square]
        square = lsb(bb_square)
        opposite_or_empty = BB_ALL & ~self.occupied_co[not color]

        if bb_square & self.knights:
            return BB_KNIGHT_ATTACKS[square] & opposite_or_empty
        elif bb_square & self.kings:
            return BB_KING_ATTACKS[square] & opposite_or_empty
        else:
            attacks = 0
            if bb_square & self.bishops or bb_square & self.queens:
                attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
            if bb_square & self.rooks or bb_square & self.queens:
                attacks |= (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
                )
            return attacks & opposite_or_empty

    def attacked_mask(self, color: bool, bb_square: Bitboard) -> Bitboard:
        """
        :param color: the color of the pieced being attacked from the bb_square
        """
        # bb_square = BB_SQUARES[square]
        square = lsb(bb_square)
        bb_opposite = self.occupied_co[color]

        if bb_square & self.pawns:
            return BB_PAWN_ATTACKS[not color][square] & bb_opposite
        elif bb_square & self.knights:
            return BB_KNIGHT_ATTACKS[square] & bb_opposite
        elif bb_square & self.kings:
            return BB_KING_ATTACKS[square] & bb_opposite
        else:
            attacks = 0
            if bb_square & self.bishops or bb_square & self.queens:
                attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
            if bb_square & self.rooks or bb_square & self.queens:
                attacks |= (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
                )
            return attacks & bb_opposite

    def defended_mask(self, color: bool, bb_square: Bitboard) -> Bitboard:
        """
        :param color: the color of the pieced being defended from the bb_square
        """
        # bb_square = BB_SQUARES[square]
        square = lsb(bb_square)
        bb_own = self.occupied_co[color]

        if bb_square & self.pawns:
            return BB_PAWN_ATTACKS[color][square] & bb_own
        elif bb_square & self.knights:
            return BB_KNIGHT_ATTACKS[square] & bb_own
        elif bb_square & self.kings:
            return BB_KING_ATTACKS[square] & bb_own
        else:
            attacks = 0
            if bb_square & self.bishops or bb_square & self.queens:
                attacks = BB_DIAG_ATTACKS[square][BB_DIAG_MASKS[square] & self.occupied]
            if bb_square & self.rooks or bb_square & self.queens:
                attacks |= (
                    BB_RANK_ATTACKS[square][BB_RANK_MASKS[square] & self.occupied]
                    | BB_FILE_ATTACKS[square][BB_FILE_MASKS[square] & self.occupied]
                )
            return attacks & bb_own

    def num_attackers(self, color: bool, square: Square) -> int:
        mask = self.attackers_mask(color, square)
        return self.get_bit_count(mask)

    def mobility(self, square: Square, color: bool) -> int:
        mask = self.mobility_mask(color, square)
        return self.get_bit_count(mask)

    def get_attacker_mobility(self, bb_square: Bitboard, capturable_color: bool):
        """
        :param capturable_color: this color piece attack will be counted as a legal move
        """

        if bb_square & self.pawns:
            return 0

        return self.mobility(bb_square, capturable_color)

    def get_mobility_delta(self, move: Move, captured_piece_type: int) -> int:
        # find mobility delta of pieces that were attacking the to_square, except pawns
        white_to_attackers_mask = self.attackers_mask(True, move.to_square)
        black_to_attackers_mask = self.attackers_mask(False, move.to_square)

        # find mobility delta of pieces that were attacking the from_square, except pawns
        white_from_attackers_mask = self.attackers_mask(True, move.from_square)
        black_from_attackers_mask = self.attackers_mask(False, move.from_square)

        # mobility of all pieces that attacked to_square
        white_to_mobility_before = sum(
            [
                self.get_attacker_mobility(attacker_bb, False)
                for attacker_bb in self.scan_bb_forward(
                    white_to_attackers_mask & ~BB_SQUARES[move.from_square]
                )
            ]
        )
        black_to_mobility_before = sum(
            [
                self.get_attacker_mobility(attacker_bb, True)
                for attacker_bb in self.scan_bb_forward(
                    black_to_attackers_mask & ~BB_SQUARES[move.from_square]
                )
            ]
        )
        # mobility of all pieces that attacked from_square
        white_from_mobility_before = sum(
            [
                self.get_attacker_mobility(attacker_bb, False)
                for attacker_bb in self.scan_bb_forward(white_from_attackers_mask)
            ]
        )
        black_from_mobility_before = sum(
            [
                self.get_attacker_mobility(attacker_bb, True)
                for attacker_bb in self.scan_bb_forward(black_from_attackers_mask)
            ]
        )

        # find mobility delta of the moving piece
        mobility_before = self.get_attacker_mobility(
            BB_SQUARES[move.from_square], not self.turn
        )

        # self.push(move)
        board_state = self.light_push(move)

        mobility_after = self.get_attacker_mobility(
            BB_SQUARES[move.to_square], self.turn
        )

        # mobility of all pieces that attacked to_square - after the move
        white_to_mobility_after = sum(
            [
                self.get_attacker_mobility(attacker_bb, False)
                for attacker_bb in self.scan_bb_forward(
                    white_to_attackers_mask & ~BB_SQUARES[move.from_square]
                )
            ]
        )
        black_to_mobility_after = sum(
            [
                self.get_attacker_mobility(attacker_bb, True)
                for attacker_bb in self.scan_bb_forward(
                    black_to_attackers_mask & ~BB_SQUARES[move.from_square]
                )
            ]
        )
        # mobility of all pieces that attacked from_square - after the move
        white_from_mobility_after = sum(
            [
                self.get_attacker_mobility(attacker_bb, False)
                for attacker_bb in self.scan_bb_forward(white_from_attackers_mask)
            ]
        )
        black_from_mobility_after = sum(
            [
                self.get_attacker_mobility(attacker_bb, True)
                for attacker_bb in self.scan_bb_forward(black_from_attackers_mask)
            ]
        )

        # pawn_mobility_after = self.get_pawn_mobility()

        # self.pop()
        self.light_pop(board_state)

        # pawn_mobility_before = self.get_pawn_mobility()

        # pawn_mobility_change = pawn_mobility_after - pawn_mobility_before

        # pawn mobility change  TODO: WRITE MORE UNIT TESTS, STILL IS NOT SAME RESULT AS WITH ABOVE CORRECT SOLUTION
        pawn_mobility_change = self.get_pawn_mobility_delta(move, captured_piece_type)

        mobility_delta = (
            mobility_after - mobility_before
            if self.turn
            else mobility_before - mobility_after
        )

        mobility_delta += (
            (white_to_mobility_after - white_to_mobility_before)
            - (black_to_mobility_after - black_to_mobility_before)
            + (white_from_mobility_after - white_from_mobility_before)
            - (black_from_mobility_after - black_from_mobility_before)
            + pawn_mobility_change
        )

        # add mobility delta caused by the capture
        if captured_piece_type:
            captured_piece_mobility = self.get_attacker_mobility(
                move.to_square, self.turn
            )
            if self.turn:
                mobility_delta += captured_piece_mobility
            else:
                mobility_delta -= captured_piece_mobility

        return mobility_delta

    @staticmethod
    def lsb(bb: Bitboard) -> int:
        return (bb & -bb).bit_length() - 1

    @staticmethod
    def scan_forward(bb: Bitboard) -> Iterator[Square]:
        while bb:
            r = bb & -bb
            yield r.bit_length() - 1
            bb ^= r

    @staticmethod
    def scan_bb_forward(bb: Bitboard) -> Iterator[Square]:
        while bb:
            r = bb & -bb
            yield r
            bb ^= r

    def get_pawn_mobility_delta(self, move: Move, captured_piece_type: int) -> int:
        pawn_mobility_change = 0

        moving_piece_type = self.piece_type_at(move.from_square)

        bb_to_square = BB_SQUARES[move.to_square]
        bb_from_square = BB_SQUARES[move.from_square]

        bb_to_up = shift_up(bb_to_square)
        bb_to_2up = shift_2_up(bb_to_square)
        bb_to_down = shift_down(bb_to_square)
        bb_to_2down = shift_2_down(bb_to_square)
        bb_from_up = shift_up(bb_from_square)
        bb_from_2up = shift_2_up(bb_from_square)
        bb_from_down = shift_down(bb_from_square)
        bb_from_2down = shift_2_down(bb_from_square)

        square_to_up = self.lsb(bb_to_up)
        square_to_2up = self.lsb(bb_to_2up)
        square_to_down = self.lsb(bb_to_down)
        square_to_2down = self.lsb(bb_to_2down)
        square_from_up = self.lsb(bb_from_up)
        square_from_2up = self.lsb(bb_from_2up)
        square_from_down = self.lsb(bb_from_down)
        square_from_2down = self.lsb(bb_from_2down)

        piece_at_square_to_up = (
            self.piece_type_at(square_to_up) if square_to_up & self.occupied else None
        )
        piece_at_square_to_2up = (
            self.piece_type_at(square_to_2up) if square_to_2up & self.occupied else None
        )
        piece_at_square_to_down = (
            self.piece_type_at(square_to_down)
            if square_to_down & self.occupied
            else None
        )
        piece_at_square_to_2down = (
            self.piece_type_at(square_to_2down)
            if square_to_2down & self.occupied
            else None
        )
        piece_at_square_from_up = (
            self.piece_type_at(square_from_up)
            if square_from_up & self.occupied
            else None
        )
        piece_at_square_from_2up = (
            self.piece_type_at(square_from_2up)
            if square_from_2up & self.occupied
            else None
        )
        piece_at_square_from_down = (
            self.piece_type_at(square_from_down)
            if square_from_down & self.occupied
            else None
        )
        piece_at_square_from_2down = (
            self.piece_type_at(square_from_2down)
            if square_from_2down & self.occupied
            else None
        )

        if (
            piece_at_square_to_up == PAWN
            and not self.color_at(square_to_up)
            and not move.from_square == square_to_up
            and not captured_piece_type
        ):
            # black pawn blocked
            if square_to_up >= 48 and piece_at_square_to_down is None:
                pawn_mobility_change += 2
            else:
                pawn_mobility_change += 1
        elif (
            square_to_2up >= 48
            and square_to_2up != move.from_square
            and piece_at_square_to_up is None
            and piece_at_square_to_2up == PAWN
            and not self.color_at(square_to_2up)
            and not captured_piece_type
        ):
            pawn_mobility_change += 1

        if (
            piece_at_square_to_down == PAWN
            and self.color_at(square_to_down)
            and not move.from_square == square_to_down
            and not captured_piece_type
        ):
            # white pawn blocked
            if square_to_down <= 16 and piece_at_square_to_up is None:
                pawn_mobility_change -= 2
            else:
                pawn_mobility_change -= 1
        elif (
            square_to_2down <= 16
            and square_to_2down != move.from_square
            and piece_at_square_to_down is None
            and piece_at_square_to_2down == PAWN
            and not self.color_at(square_to_2down)
            and not captured_piece_type
        ):
            pawn_mobility_change -= 1

        if piece_at_square_from_up == PAWN and not self.color_at(square_from_up):
            # black pawn unblocked
            if (
                square_from_up >= 48
                and square_from_down != move.to_square
                and piece_at_square_from_down is None
            ):
                pawn_mobility_change -= 2
            else:
                pawn_mobility_change -= 1
        elif (
            square_from_2up >= 48
            and piece_at_square_from_up is None
            and piece_at_square_from_2up == PAWN
            and not self.color_at(square_from_2up)
        ):
            pawn_mobility_change -= 1

        if piece_at_square_from_down == PAWN and self.color_at(square_from_down):
            # white pawn unblocked
            if (
                square_from_down <= 16
                and square_from_up != move.to_square
                and self.piece_type_at(square_from_up) is None
            ):
                pawn_mobility_change += 2
            else:
                pawn_mobility_change += 1
        elif (
            square_from_2down <= 16
            and piece_at_square_from_down is None
            and piece_at_square_from_2down == PAWN
            and not self.color_at(square_from_2down)
        ):
            pawn_mobility_change += 1

        # pawn has moved from first rank losing additional 1 mobility
        if moving_piece_type == PAWN:
            if move.from_square <= 16 and self.turn:
                pawn_mobility_change -= 1
            elif move.from_square >= 48 and not self.turn:
                pawn_mobility_change += 1

            # blocked own next move
            if (
                self.turn
                and piece_at_square_from_up is None
                and piece_at_square_to_up is not None
            ):
                pawn_mobility_change -= 1
            elif (
                not self.turn
                and piece_at_square_from_down is None
                and self.piece_type_at(square_to_down) is not None
            ):
                pawn_mobility_change += 1
            # unblocked itself by capturing
            elif (
                self.turn
                and piece_at_square_from_up is not None
                and piece_at_square_to_up is None
            ):
                pawn_mobility_change += 1
            elif (
                not self.turn
                and piece_at_square_from_down is not None
                and piece_at_square_to_down is None
            ):
                pawn_mobility_change -= 1

        # pawn was captured that could have moved
        if captured_piece_type == PAWN:
            # black pawn captured
            if self.turn and piece_at_square_to_down is None:
                if move.to_square >= 48 and piece_at_square_to_2down is None:
                    pawn_mobility_change += 2
                else:
                    pawn_mobility_change += 1
            # white pawn captured
            elif not self.turn and piece_at_square_to_up is None:
                if move.to_square <= 16 and piece_at_square_to_2up is None:
                    pawn_mobility_change -= 2
                else:
                    pawn_mobility_change -= 1

        return pawn_mobility_change

    def get_total_mobility(self, turn: bool) -> Tuple[int, int]:
        king_square = self.king(turn)
        original_turn = self.turn
        if original_turn != turn:
            self.turn = not self.turn

        mobilty = 0
        king_mobility = 0
        # for move, is_legal in self.generate_moves_with_legal_flag(king_square):
        for move in self.generate_pseudo_legal_moves_no_castling():
            # if is_legal:
            #     mobilty += 0.5
            # mobilty += 0.5
            mobilty += 1
            if move.from_square == king_square:
                king_mobility += 1

        if original_turn != turn:  # turn back for safety against mutable board passed
            self.turn = not self.turn

        return mobilty, king_mobility

    def get_king_threats(self, color: bool):
        king_square = self.king(not color)
        assert king_square is not None
        king_proximity_squares = self.attacks(king_square)
        king_threats = 0
        for square in king_proximity_squares:
            mask = self.attackers_mask(color, square)
            king_threats += self.get_bit_count(mask)
        return king_threats

    def generate_moves_with_legal_flag(
        self, king_square: Square, from_mask=BB_ALL, to_mask=BB_ALL
    ) -> Generator:
        blockers = self._slider_blockers(king_square)
        checkers = self.attackers_mask(not self.turn, king_square)
        moves_generated = set()
        if checkers:
            for move in self._generate_evasions(
                king_square, checkers, from_mask, to_mask
            ):
                if self._is_safe(king_square, blockers, move):
                    moves_generated.add(move)
                    yield move, True
                else:
                    moves_generated.add(move)
                    yield move, False
        for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
            if move not in moves_generated:
                if self._is_safe(king_square, blockers, move):
                    yield move, True
                else:
                    yield move, False

    def is_empty(self, square: Square) -> Bitboard:
        mask = BB_SQUARES[square]
        return not self.occupied & mask

    def len_empty_squares_around_king(self, color: bool, move: Move) -> int:
        king_move = False
        king_square = self.king(color)
        if not king_square:
            return 0

        if move.from_square == king_square:
            self.push(move)
            king_square = move.to_square
            king_move = True

        king_mobiliity = len(
            [square for square in self.attacks(king_square) if self.is_empty(square)]
        )

        if king_move:
            self.pop()

        return king_mobiliity

    # def len_empty_squares_around_king(self, color: bool, move: Move) -> int:
    #     king_move = False
    #     king_square = self.king(color)
    #     if not king_square:
    #         return 0
    #
    #     if move.from_square == king_square:
    #         state = self.light_push(move)
    #         king_square = move.to_square
    #         king_move = True
    #
    #     king_mobiliity = self.get_bit_count(self.attacks_mask(king_square) & ~self.occupied)
    #
    #     if king_move:
    #         self.light_pop(state)
    #
    #     return king_mobiliity

    def get_pawn_mobility(self) -> int:
        original_turn = self.turn
        self.turn = True
        white_pawn_mobility = len([move for move in self.generate_pawn_moves()])
        self.turn = False
        black_pawn_mobility = len([move for move in self.generate_pawn_moves()])
        self.turn = original_turn
        return white_pawn_mobility - black_pawn_mobility

    def generate_pawn_moves(self) -> Generator:
        pawns = self.pawns & self.occupied_co[self.turn] & BB_ALL
        if not pawns:
            return

        # Prepare pawn advance generation.
        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        single_moves &= BB_ALL
        double_moves &= BB_ALL

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield Move(from_square, to_square)

    def generate_pseudo_legal_moves_no_castling(
        self, from_mask=BB_ALL, to_mask=BB_ALL
    ) -> Generator:
        our_pieces = self.occupied_co[self.turn]

        # Generate piece moves.
        non_pawns = our_pieces & ~self.pawns & from_mask
        for from_square in scan_reversed(non_pawns):
            moves = self.attacks_mask(from_square) & ~our_pieces & to_mask
            for to_square in scan_reversed(moves):
                yield Move(from_square, to_square)

        # The remaining moves are all pawn moves.
        pawns = self.pawns & self.occupied_co[self.turn] & from_mask
        if not pawns:
            return

        # Generate pawn captures.
        capturers = pawns
        for from_square in scan_reversed(capturers):
            targets = (
                BB_PAWN_ATTACKS[self.turn][from_square]
                & self.occupied_co[not self.turn]
                & to_mask
            )

            for to_square in scan_reversed(targets):
                if square_rank(to_square) in [0, 7]:
                    yield Move(from_square, to_square, QUEEN)
                    yield Move(from_square, to_square, ROOK)
                    yield Move(from_square, to_square, BISHOP)
                    yield Move(from_square, to_square, KNIGHT)
                else:
                    yield Move(from_square, to_square)

        # Prepare pawn advance generation.
        if self.turn == WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = single_moves << 8 & ~self.occupied & (BB_RANK_3 | BB_RANK_4)
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = single_moves >> 8 & ~self.occupied & (BB_RANK_6 | BB_RANK_5)

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, QUEEN)
                yield Move(from_square, to_square, ROOK)
                yield Move(from_square, to_square, BISHOP)
                yield Move(from_square, to_square, KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == BLACK else -16)
            yield Move(from_square, to_square)

        # Generate en passant captures.
        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def get_moved_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    def get_moving_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.from_square) or 0

    def get_captured_piece_type(self, move: Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    @staticmethod
    def is_on_border(square, rank=None) -> bool:
        return square_file(square) in [0, 7] or (rank or square_rank(square)) in [0, 7]

    @staticmethod
    def get_piece_value(
        piece_type, color, square=None, rank=0, n_pawns=8, bishop_pair=0
    ) -> float:
        if not piece_type:
            return 0

        if square is not None and rank == 0 and piece_type == PAWN:
            rank = square_rank(square) if color else 7 - square_rank(square)

        if piece_type == PAWN:
            return 1 + pow((rank / 5), 6)
        if piece_type == KNIGHT:
            return 3
        if piece_type == BISHOP:
            return 3 + bishop_pair  # * (8 - n_pawns) / 8
        if piece_type == ROOK:
            return 4.5
        if piece_type == QUEEN:
            return 9

        return 0

    @staticmethod
    def get_fen_opposite_turn(fen: str) -> str:
        return fen.replace(" w ", " b ") if " w " in fen else fen.replace(" b ", " w ")
