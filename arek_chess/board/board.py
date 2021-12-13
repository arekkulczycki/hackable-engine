# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

from typing import Tuple

import chess
import numpy
from chess import Board as ChessBoard, scan_reversed, Move, square_rank


class Board(ChessBoard):
    """
    Class_docstring
    """

    def get_score_params(self, color):
        material, safety, under_attack = self.get_material_and_safety(color)

        mobility, king_mobility = self.get_total_mobility(color)

        return material, safety, under_attack, mobility, king_mobility

    def get_score(self, action, moved_piece_type):
        white_params = numpy.array(
            [*self.get_score_params(True)],
            dtype=numpy.float32,
        )
        black_params = numpy.array(
            [*self.get_score_params(False)],
            dtype=numpy.float32,
        )
        score = self.calculate_score(
            action, white_params - black_params, moved_piece_type
        )

        return round(score, 3)

    def get_score_from_params(self, action, moved_piece_type, params):
        score = self.calculate_score(action, params, moved_piece_type)

        return round(score, 3)

    def get_shallow_score(self, action, piece_type=None):
        """"""

    @staticmethod
    def calculate_score(action, params, piece_type=None):
        # TODO: when no pieces on board, transform action params of pieces to columns of pawns to use! train pawn endgames separately
        piece_type_bonus = (
            0  # 10.0 * action[piece_type - 1 + 4] if piece_type is not None else 0
        )
        return numpy.dot(action, params) + piece_type_bonus

    def get_material_delta(self, captured_piece_type):
        # if white to move then capture is + in material, otherwise is -
        sign = 1 if self.turn else -1

        # color of the captured piece is opposite to turn (side making the move)
        return self.get_piece_value(captured_piece_type, not self.turn) * sign

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

        def get_safety(square):
            """ """
            # TODO: optimize somehow, with functools or numba?
            safety = 0
            for attacked_square in self.attacks(square):
                piece_color = self.color_at(attacked_square)
                if piece_color == color:
                    piece_type = self.piece_type_at(attacked_square)
                    if piece_type != chess.KING:
                        safety += self.get_piece_value(piece_type, color)
            return safety

        # if move color was the same as calculated color (self.turn == color) then capture doesn't matter
        if self.turn == color:
            # TODO: potentially take len below without initializing SquareSet objects?

            # safety coming from protecting other squares by the moved piece, before the move
            safety_before = get_safety(move.from_square)

            # safety coming from the moved piece being protected, before
            if moved_piece_type != chess.KING:
                safety_before += len(
                    self.attackers(color, move.from_square)
                ) * self.get_piece_value(moved_piece_type, color)

            self.push(move)

            # safety coming from protecting other squares by the moved piece, after
            safety_after = get_safety(move.to_square)

            # safety coming from the moved piece being protected, after
            if moved_piece_type != chess.KING:
                safety_after += len(
                    self.attackers(color, move.to_square)
                ) * self.get_piece_value(moved_piece_type, color)

            self.pop()

            return safety_after - safety_before

        else:  # moving side is opposite to calculated safety
            # in this case the safety change is:
            #  - if captured, equal to all attacks on the captured piece

            if captured_piece_type:
                safety_loss = len(
                    self.attackers(color, move.to_square)
                ) * self.get_piece_value(captured_piece_type, color)

                for square in self.attacks(move.to_square):
                    piece_color = self.color_at(square)
                    if piece_color == color:
                        piece_type = self.piece_type_at(square)
                        if piece_type:
                            safety_loss += self.get_piece_value(piece_type, color)

                return -safety_loss

            #  - otherwise, if piece got in the way of protectors, for all the attackers on the current piece
            #    must be found if they protected a piece in previous position

            safety_loss = 0

            attackers_before = self.attackers(color, move.to_square)

            self.push(move)

            attackers_after = self.attackers(color, move.to_square)

            #  - or opposite, piece went out of the way of protectors
            for attacker in attackers_before:
                for square in self.attacks(attacker):
                    piece_color = self.color_at(square)
                    if piece_color == color:
                        piece_type = self.piece_type_at(square)
                        if piece_type:
                            safety_loss -= self.get_piece_value(piece_type, piece_color)

            self.pop()

            for attacker in attackers_after:
                for square in self.attacks(attacker):
                    piece_color = self.color_at(square)
                    if piece_color == color:
                        piece_type = self.piece_type_at(square)
                        if piece_type:
                            safety_loss += self.get_piece_value(piece_type, piece_color)

            return -safety_loss

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

        def get_under_attack(square, _color=color):
            """
            Return sum value of all attacked pieces from square of _color
            """
            # TODO: optimize somehow, with functools or numba?
            under_attack = 0
            for attacked_square in self.attacks(square):
                piece_color = self.color_at(attacked_square)
                if piece_color == _color:
                    piece_type = self.piece_type_at(attacked_square)
                    if piece_type:
                        under_attack += self.get_piece_value(piece_type, _color)
            return under_attack

        # if calculating same color as making the move
        if color == self.turn:
            attackers_before = self.attackers(not self.turn, move.from_square)
            self.push(move)
            attackers_after = self.attackers(self.turn, move.to_square)
            self.pop()

            # find sum of attacks value before the move
            under_attack_before = 0
            for attacker in attackers_before:
                under_attack_before += get_under_attack(attacker)
            for attacker in attackers_after:
                under_attack_before += get_under_attack(attacker)

            self.push(move)

            # find sum of attacks value after the move
            under_attack_after = 0
            for attacker in attackers_before:
                under_attack_after += get_under_attack(attacker)
            for attacker in attackers_after:
                under_attack_after += get_under_attack(attacker)

            self.pop()

            return under_attack_after - under_attack_before

        # calculating for opposition side
        else:
            attacks_before = get_under_attack(
                move.from_square
            )  # self.attacks(move.from_square)
            self.push(move)
            blocked_attackers = self.attackers(not self.turn, move.to_square)

            # minus all the blocked attacks from previous position
            blocked_attacks_after = 0
            for attacker in blocked_attackers:
                if self.piece_type_at(attacker) in [
                    chess.QUEEN,
                    chess.ROOK,
                    chess.BISHOP,
                ]:
                    blocked_attacks_after += get_under_attack(attacker, self.turn)

            attacks_after = get_under_attack(move.to_square)
            self.pop()

            blocked_attacks_before = 0
            for attacker in blocked_attackers:
                if self.piece_type_at(attacker) in [
                    chess.QUEEN,
                    chess.ROOK,
                    chess.BISHOP,
                ]:
                    blocked_attacks_before += get_under_attack(attacker, not self.turn)

            return (
                attacks_after
                - attacks_before
                + (blocked_attacks_after - blocked_attacks_before)
            )

    def get_material_and_safety(self, color) -> Tuple[float, float, float]:
        material = 0.0
        safety = 0.0
        under_attack = 0.0
        n_pawns = 0

        for pawn in self.pieces(piece_type=chess.PAWN, color=color):
            n_pawns += 1
            rank = chess.square_rank(pawn) if color else 7 - chess.square_rank(pawn)
            val = self.get_piece_value(chess.PAWN, color, rank=rank)
            material += val

            attackers = self.attackers(color, pawn)
            for _ in attackers:
                safety += val
            for _ in self.attackers(not color, pawn):
                under_attack += val

        for knight in self.pieces(piece_type=chess.KNIGHT, color=color):
            # TODO: account for stage of game
            val = 3
            material += val
            for _ in self.attackers(color, knight):
                safety += val
            for _ in self.attackers(not color, knight):
                under_attack += val

        bishop_pair = 0
        for bishop in self.pieces(piece_type=chess.BISHOP, color=color):
            val = self.get_piece_value(
                chess.BISHOP, color, n_pawns=n_pawns, bishop_pair=bishop_pair
            )
            material += val
            for _ in self.attackers(color, bishop):
                safety += val
            for _ in self.attackers(not color, bishop):
                under_attack += val
            bishop_pair += 1

        for rook in self.pieces(piece_type=chess.ROOK, color=color):
            val = 4.5
            material += val
            for _ in self.attackers(color, rook):
                safety += val
            for _ in self.attackers(not color, rook):
                under_attack += val

        for queen in self.pieces(piece_type=chess.QUEEN, color=color):
            val = 9
            material += val
            for _ in self.attackers(color, queen):
                safety += val
            for _ in self.attackers(not color, queen):
                under_attack += val

        return material, safety, under_attack

    def get_mobility_delta(self, move: Move, captured_piece_type) -> int:
        def get_attacker_mobility(attacker, capturable_color):
            """ 
            :param capturable_color: this color piece attack will be counted as a legal move 
            """
            bb_square = chess.BB_SQUARES[attacker]
            if bb_square & self.pawns:
                return 0

            mobility = 0
            for attack in self.attacks(attacker):
                if self.is_empty(attack) or self.color_at(attack) == capturable_color:
                    mobility += 1
            return mobility

        # find mobility delta of pieces that were attacking the to_square, except pawns
        white_to_attackers = self.attackers(True, move.to_square)
        black_to_attackers = self.attackers(False, move.to_square)

        # find mobility delta of pieces that were attacking the from_square, except pawns
        white_from_attackers = self.attackers(True, move.from_square)
        black_from_attackers = self.attackers(False, move.from_square)

        # mobility of all pieces that attacked to_square
        white_to_mobility_before = sum(
            [
                get_attacker_mobility(attacker, False)
                for attacker in white_to_attackers
                if attacker != move.from_square
            ]
        )
        black_to_mobility_before = sum(
            [
                get_attacker_mobility(attacker, True)
                for attacker in black_to_attackers
                if attacker != move.from_square
            ]
        )
        # mobility of all pieces that attacked from_square
        white_from_mobility_before = sum(
            [
                get_attacker_mobility(attacker, False)
                for attacker in white_from_attackers
            ]
        )
        black_from_mobility_before = sum(
            [
                get_attacker_mobility(attacker, True)
                for attacker in black_from_attackers
            ]
        )

        # find mobility delta of the moving piece
        mobility_before = get_attacker_mobility(move.from_square, not self.turn)

        # TODO: optimize to only look at 1 pawn
        pawn_mobility_before = self.get_pawn_mobility()

        self.push(move)

        mobility_after = get_attacker_mobility(move.to_square, self.turn)

        # mobility of all pieces that attacked to_square - after the move
        white_to_mobility_after = sum(
            [
                get_attacker_mobility(attacker, False)
                for attacker in white_to_attackers
                if attacker != move.from_square
            ]
        )
        black_to_mobility_after = sum(
            [
                get_attacker_mobility(attacker, True)
                for attacker in black_to_attackers
                if attacker != move.from_square
            ]
        )
        # mobility of all pieces that attacked from_square - after the move
        white_from_mobility_after = sum(
            [
                get_attacker_mobility(attacker, False)
                for attacker in white_from_attackers
            ]
        )
        black_from_mobility_after = sum(
            [
                get_attacker_mobility(attacker, True)
                for attacker in black_from_attackers
            ]
        )

        pawn_mobility_after = self.get_pawn_mobility()

        self.pop()

        mobility_delta = mobility_after - mobility_before if self.turn else mobility_before - mobility_after

        mobility_delta += (
            (white_to_mobility_after - white_to_mobility_before)
            - (black_to_mobility_after - black_to_mobility_before)
            + (white_from_mobility_after - white_from_mobility_before)
            - (black_from_mobility_after - black_from_mobility_before)
            + (pawn_mobility_after - pawn_mobility_before)
        )

        # add mobility delta caused by the capture
        if captured_piece_type:
            captured_piece_mobility = get_attacker_mobility(move.to_square, self.turn)
            if self.turn:
                mobility_delta += captured_piece_mobility
            else:
                mobility_delta -= captured_piece_mobility

        # print('***')
        # print(mobility_after, mobility_before)
        # print(white_to_mobility_after, white_to_mobility_before)
        # print(white_from_mobility_before, white_from_mobility_after)
        # print(black_to_mobility_after, black_to_mobility_before)
        # print(black_from_mobility_after, black_from_mobility_before)
        # print(pawn_mobility_after, pawn_mobility_before)
        # if captured_piece_type:
        #     print(captured_piece_mobility)

        return mobility_delta

    def get_total_mobility(self, turn: bool) -> Tuple[int, int]:
        # king_square = self.pieces(piece_type=chess.KING, color=turn).pop()
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

    def generate_moves_with_legal_flag(
        self, king_square, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL
    ):
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

    def is_empty(self, square):
        mask = chess.BB_SQUARES[square]
        return not self.occupied & mask

    def len_empty_squares_around_king(self, color, move: Move):
        king_move = False
        king_square = self.king(color)

        if move.from_square == king_square:
            self.push(move)
            king_square = move.to_square
            king_move = True

        king_mobiliity = len([square for square in self.attacks(king_square) if self.is_empty(square)])

        if king_move:
            self.pop()

        return king_mobiliity

    def get_pawn_mobility(self):
        original_turn = self.turn
        self.turn = True
        white_pawn_mobility = len([move for move in self.generate_pawn_moves()])
        self.turn = False
        black_pawn_mobility = len([move for move in self.generate_pawn_moves()])
        self.turn = original_turn
        return white_pawn_mobility - black_pawn_mobility

    def generate_pawn_moves(self):
        pawns = self.pawns & self.occupied_co[self.turn] & chess.BB_ALL
        if not pawns:
            return

        # Prepare pawn advance generation.
        if self.turn == chess.WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = (
                single_moves << 8 & ~self.occupied & (chess.BB_RANK_3 | chess.BB_RANK_4)
            )
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = (
                single_moves >> 8 & ~self.occupied & (chess.BB_RANK_6 | chess.BB_RANK_5)
            )

        single_moves &= chess.BB_ALL
        double_moves &= chess.BB_ALL

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == chess.BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, chess.QUEEN)
                yield Move(from_square, to_square, chess.ROOK)
                yield Move(from_square, to_square, chess.BISHOP)
                yield Move(from_square, to_square, chess.KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == chess.BLACK else -16)
            yield Move(from_square, to_square)

    def generate_pseudo_legal_moves_no_castling(
        self, from_mask=chess.BB_ALL, to_mask=chess.BB_ALL
    ):
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
                chess.BB_PAWN_ATTACKS[self.turn][from_square]
                & self.occupied_co[not self.turn]
                & to_mask
            )

            for to_square in scan_reversed(targets):
                if square_rank(to_square) in [0, 7]:
                    yield Move(from_square, to_square, chess.QUEEN)
                    yield Move(from_square, to_square, chess.ROOK)
                    yield Move(from_square, to_square, chess.BISHOP)
                    yield Move(from_square, to_square, chess.KNIGHT)
                else:
                    yield Move(from_square, to_square)

        # Prepare pawn advance generation.
        if self.turn == chess.WHITE:
            single_moves = pawns << 8 & ~self.occupied
            double_moves = (
                single_moves << 8 & ~self.occupied & (chess.BB_RANK_3 | chess.BB_RANK_4)
            )
        else:
            single_moves = pawns >> 8 & ~self.occupied
            double_moves = (
                single_moves >> 8 & ~self.occupied & (chess.BB_RANK_6 | chess.BB_RANK_5)
            )

        single_moves &= to_mask
        double_moves &= to_mask

        # Generate single pawn moves.
        for to_square in scan_reversed(single_moves):
            from_square = to_square + (8 if self.turn == chess.BLACK else -8)

            if square_rank(to_square) in [0, 7]:
                yield Move(from_square, to_square, chess.QUEEN)
                yield Move(from_square, to_square, chess.ROOK)
                yield Move(from_square, to_square, chess.BISHOP)
                yield Move(from_square, to_square, chess.KNIGHT)
            else:
                yield Move(from_square, to_square)

        # Generate double pawn moves.
        for to_square in scan_reversed(double_moves):
            from_square = to_square + (16 if self.turn == chess.BLACK else -16)
            yield Move(from_square, to_square)

        # Generate en passant captures.
        if self.ep_square:
            yield from self.generate_pseudo_legal_ep(from_mask, to_mask)

    def get_moved_piece_type(self, move: chess.Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    def get_moving_piece_type(self, move: chess.Move) -> int:
        return self.piece_type_at(move.from_square) or 0

    def get_captured_piece_type(self, move: chess.Move) -> int:
        return self.piece_type_at(move.to_square) or 0

    @staticmethod
    def is_on_border(square, rank=None):
        return chess.square_file(square) in [0, 7] or (
            rank or chess.square_rank(square)
        ) in [0, 7]

    @staticmethod
    def get_piece_value(piece, color, square=None, rank=0, n_pawns=8, bishop_pair=0):
        if not piece:
            return 0

        if isinstance(piece, int):
            piece_type = piece
        else:
            piece_type = piece.piece_type

        if square is not None and rank == 0 and piece_type == chess.PAWN:
            rank = chess.square_rank(square) if color else 7 - chess.square_rank(square)

        if piece_type == chess.PAWN:
            return 1 + pow((rank / 5), 6)
        if piece_type == chess.KNIGHT:
            return 3
        if piece_type == chess.BISHOP:
            return 3 + bishop_pair  # * (8 - n_pawns) / 8
        if piece_type == chess.ROOK:
            return 4.5
        if piece_type == chess.QUEEN:
            return 9
        if piece_type == chess.KING:  # TODO: is that right?
            return 0

    @staticmethod
    def get_fen_opposite_turn(fen):
        return fen.replace(" w ", " b ") if " w " in fen else fen.replace(" b ", " w ")
