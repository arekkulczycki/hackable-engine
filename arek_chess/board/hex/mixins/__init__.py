# -*- coding: utf-8 -*-
from typing import Dict, Protocol

from arek_chess.board import BitBoard


class HexBoardProtocol(Protocol):
    turn: bool
    occupied_co: Dict[bool, BitBoard]


class BoardShapeError(Exception):
    """"""
