# -*- coding: utf-8 -*-
from typing import List, Protocol


class ChessBoardProtocol(Protocol):
    turn: bool
    pawns: int
    knights: int
    bishops: int
    rooks: int
    queens: int
    kings: int
    occupied_co: List[int]
    castling_rights: int
    ep_square: int
