# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC
from typing import Any, List, Optional, Union

BitBoard = int
AnyMove = Any


class GameMoveBase(ABC):
    """"""

    def __init__(self, *args, **kwargs) -> None:
        """"""


class GameBoardBase(ABC):
    """"""

    turn: bool
    move_stack: List
    legal_moves: Any
    has_draws: bool = True
    has_move_limit: bool = False
    """Some games can be played forever, others are limited like for instance Hex: up until the board is filled."""

    def __init__(self, *args, **kwargs) -> None:
        """"""

    def position(self) -> str:
        """"""

        raise NotImplementedError

    def copy(self) -> Any:
        """"""

        raise NotImplementedError

    def push(self, move: AnyMove) -> None:
        """"""

        raise NotImplementedError

    def push_coord(self, coord: str) -> None:
        """"""

        raise NotImplementedError

    def pop(self) -> AnyMove:
        """"""

        raise NotImplementedError

    def is_game_over(self, *, claim_draw: bool = False) -> bool:
        """"""

        raise NotImplementedError

    def winner(self) -> Optional[bool]:
        """"""

        raise NotImplementedError

    def get_forcing_level(self, move: AnyMove) -> int:
        """"""

        raise NotImplementedError

    def is_check(self) -> bool:
        """"""

        raise NotImplementedError

    def serialize_position(self) -> bytes:
        """"""

        raise NotImplementedError

    def deserialize_position(self, bytes_: bytes) -> None:
        """"""

        raise NotImplementedError

    def get_notation(self) -> str:
        """"""

        raise NotImplementedError
