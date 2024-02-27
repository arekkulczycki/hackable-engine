# -*- coding: utf-8 -*-
from typing import Any, NamedTuple, Optional, Type

from hackable_engine.board import GameBoardBase


class EvalWorkerConfig(NamedTuple):
    """
    Definition of required and optional arguments for EvalWorker initialization.
    """

    worker_number: int
    board_class: Type[GameBoardBase]
    board_size: Optional[int]
    is_training_run: bool = False
    ai_model: Optional[Any] = None
