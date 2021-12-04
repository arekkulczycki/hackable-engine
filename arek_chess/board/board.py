# -*- coding: utf-8 -*-
"""
Module_docstring.
"""

import chess
import numpy
from chess import Board as ChessBoard


class Board(ChessBoard):
    """
    Class_docstring
    """

    def get_score(self, action, piece_type=None):
        white_params = numpy.array(
            [
                *self.get_material_and_safety(True),
                *self.get_total_mobility(True),
            ],
            dtype=numpy.float32,
        )
        black_params = numpy.array(
            [
                *self.get_material_and_safety(False),
                *self.get_total_mobility(False),
            ],
            dtype=numpy.float32,
        )
        score = self.get_cached_score(action, white_params, black_params, piece_type)

        return score

    @staticmethod
    # @njit
    def get_cached_score(action, white_params, black_params, piece_type=None):
        # TODO: when no pieces on board, transform action params of pieces to columns of pawns to use! train pawn endgames separately
        piece_type_bonus = 10.0 * action[piece_type - 1 + 4] if piece_type is not None else 0
        return action[0] * (white_params[0] - black_params[0]) + \
               action[1] * (white_params[1] - black_params[1]) + \
               action[2] * (white_params[2] - black_params[2]) + \
               action[3] * (white_params[3] - black_params[3]) + \
               piece_type_bonus

    def get_material_and_safety(self, is_white):
        material = 0.0
        safety = 0.0
        n_pawns = 0

        for pawn in self.pieces(piece_type=chess.PAWN, color=is_white):
            n_pawns += 1
            rank = chess.square_rank(pawn) if is_white else 7 - chess.square_rank(pawn)
            val = self.get_piece_value(chess.PAWN, is_white, rank=rank)
            material += val
            if self.is_attacked_by(is_white, pawn):
                safety += val
            if self.is_on_border(pawn, rank):
                safety += val / 2

        for knight in self.pieces(piece_type=chess.KNIGHT, color=is_white):
            # TODO: account for stage of game
            val = 3
            material += val
            if self.is_attacked_by(is_white, knight):
                safety += val
            if self.is_on_border(knight):
                safety += val / 3

        bishop_pair = 0
        for bishop in self.pieces(piece_type=chess.BISHOP, color=is_white):
            val = self.get_piece_value(
                chess.BISHOP, is_white, n_pawns=n_pawns, bishop_pair=bishop_pair
            )
            material += val
            if self.is_attacked_by(is_white, bishop):
                safety += val
            if self.is_on_border(bishop):
                safety += val / 2
            bishop_pair += 1.5

        for rook in self.pieces(piece_type=chess.ROOK, color=is_white):
            val = 4.5
            material += val
            if self.is_attacked_by(is_white, rook):
                safety += val
            if self.is_on_border(rook):
                safety += val / 3

        for queen in self.pieces(piece_type=chess.QUEEN, color=is_white):
            val = 9
            material += val
            if self.is_attacked_by(is_white, queen):
                safety += val
            if self.is_on_border(queen):
                safety += val / 2

        return material, safety

    def get_total_mobility(self, for_white):
        king_square = self.pieces(piece_type=chess.KING, color=for_white).pop()
        if self.turn != for_white:
            self.turn = not self.turn

        average_move_count = 0
        king_mobility = 0
        for move, is_legal in self.generate_moves_with_legal_flag(king_square):
            if is_legal:
                average_move_count += 0.5
            average_move_count += 0.5
            if move.from_square == king_square:
                king_mobility += 1

        if self.turn != for_white:  # turn back for safety against mutable board passed
            self.turn = not self.turn

        return average_move_count, king_mobility

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

    def get_moving_piece_type(self, move):
        return self.piece_type_at(move.from_square)

    @staticmethod
    def is_on_border(square, rank=None):
        return chess.square_file(square) in [0, 7] or (
            rank or chess.square_rank(square)
        ) in [0, 7]

    @staticmethod
    def get_piece_value(piece, is_white, square=None, rank=0, n_pawns=8, bishop_pair=0):
        if not piece:
            return 0

        if isinstance(piece, int):
            piece_type = piece
        else:
            piece_type = piece.piece_type

        if square is not None and rank == 0 and piece_type == chess.PAWN:
            rank = (
                chess.square_rank(square) if is_white else 7 - chess.square_rank(square)
            )

        if piece_type == chess.PAWN:
            return 1 + pow((rank / 5), 6)
        if piece_type == chess.KNIGHT:
            return 3
        if piece_type == chess.BISHOP:
            return 3 + bishop_pair * (8 - n_pawns) / 8
        if piece_type == chess.ROOK:
            return 4.5
        if piece_type == chess.QUEEN:
            return 9
        if piece_type == chess.KING:  # TODO: is that right?
            return 0
