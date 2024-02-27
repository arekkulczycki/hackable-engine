"""
Evaluation by all the attributes obtained from board in an optimized way.

Desired:
[x] is_check
[x] material (with advanced pawn bonus)
[x] mobility
[x] threats (x ray)
[x] king threats (x ray)
[x] direct threats (knight and pawn attacks)
[x] king safety/mobility (?)
[x] color on which white pieces are
[x] color on which black pieces are
[x] color on which white pawns are
[x] color on which black pawns are
[x] protection
[ ] king proximity threats (direct)
[ ] advancement
[ ] protection x advancement
[ ] pawn structure defined as a binary number
[ ]

Observation:
[x] material on board
[x] white king location
[x] black king location
[x] white forces location (density of attacks, 64 floats or simplified to just avg coordinates)
[x] black forces location (density of attacks, 64 floats or simplified to just avg coordinates)
[x] colors of remaining white bishops
[x] colors of remaining black bishops
[ ] openness of the position (many pawns locked > ... > many pawns gone)

?
[x] queens on board (at least 1) - not clearly important given material is already measured
[x] pawns on board (how many)
[ ]
"""

from typing import Tuple, Optional

from numpy import float32, dot

from hackable_engine.board.board import Board
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval

ACTION_TYPE = Tuple[
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
    double,
]


class MultiIdeaEval(BaseEval):
    """"""

    DEFAULT_ACTION: WeightsType = (
        double(10.0),  # is_check
        double(100.0),  # material
        double(5.0),  # mobility
        double(1.0),  # threats
        double(2.0),  # king threats
        double(2.0),  # direct threats
        double(-10.0),  # king safety/mobility
        double(3.0),  # protection
        double(0.0),  # light_pieces_white
        double(0.0),
        double(0.0),
        double(0.0),
        double(0.0),  # light_pawns_white
        double(0.0),
        double(0.0),
        double(0.0),
    )
    PARAMS_NUMBER: int = 16

    def get_score(
        self,
        board: Board,
        move_str: str,
        captured_piece_type: int,
        is_check: bool,
        action: Optional[WeightsType] = None,
    ) -> double:
        """"""

        if action is None:
            action = self.DEFAULT_ACTION

        is_check_int = (
            -int(is_check) if board.turn else int(is_check)
        )  # color is the one who gave the check
        material = (
            board.get_material_no_pawns(True)
            + board.get_material_pawns(True)
            - board.get_material_no_pawns(False)
            - board.get_material_pawns(False)
        )
        mobility = board.get_mobility(True) - board.get_mobility(False)
        threats = board.get_threats(True) - board.get_threats(False)
        king_threats = board.get_king_threats(True) - board.get_king_threats(False)
        direct_threats = board.get_direct_threats(True) - board.get_direct_threats(
            False
        )
        king_mobility = board.get_king_mobility(True) - board.get_king_mobility(False)
        protection = board.get_protection(True) - board.get_protection(False)

        light_pieces_white = board.pieces_on_light_squares(True)
        dark_pieces_white = board.pieces_on_dark_squares(True)
        light_pieces_black = board.pieces_on_light_squares(False)
        dark_pieces_black = board.pieces_on_dark_squares(False)

        light_pawns_white = board.pawns_on_light_squares(True)
        dark_pawns_white = board.pawns_on_dark_squares(True)
        light_pawns_black = board.pawns_on_light_squares(False)
        dark_pawns_black = board.pawns_on_dark_squares(False)

        params = (
            double(is_check_int),
            double(material),
            double(mobility),
            double(threats),
            double(king_threats),
            double(direct_threats),
            double(king_mobility),
            double(protection),
            double(light_pieces_white),
            double(light_pieces_black),
            double(dark_pieces_white),
            double(dark_pieces_black),
            double(light_pawns_white),
            double(light_pawns_black),
            double(dark_pawns_white),
            double(dark_pawns_black),
        )

        return self.calculate_score(action, params)

    @staticmethod
    def calculate_score(
        action: WeightsType, params: WeightsType
    ) -> double:
        return dot(action, params)
