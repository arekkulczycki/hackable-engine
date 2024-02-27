# -*- coding: utf-8 -*-
from typing import Dict, List, Protocol, Tuple

from hackable_engine.board import BitBoard


class HexBoardProtocol(Protocol):
    size: int
    size_square: int
    turn: bool
    occupied_co: Dict[bool, BitBoard]
    unoccupied: BitBoard

    board_bytes_number: int

    def _board_bytes_number(self, size_square: int) -> int: ...
    def serialize_position(self) -> bytes: ...
    def deserialize_position(self, bytes_: bytes) -> None: ...
    def get_all_mask(self) -> BitBoard: ...
    def get_occupied_in_components(self, color: bool) -> Tuple[BitBoard]: ...
    def split_bitboard_in_two_components(self, b: BitBoard) -> Tuple[BitBoard, BitBoard]: ...
    def split_bitboard_in_three_components(self, b: BitBoard) -> Tuple[BitBoard, BitBoard, BitBoard]: ...
    def unpack_components(self, bytes_: bytes) -> BitBoard: ...


class BoardShapeError(Exception):
    """"""
