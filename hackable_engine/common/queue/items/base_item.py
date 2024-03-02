# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Self


@dataclass
class BaseItem(ABC):
    """
    Item passed through DispatcherQueue.
    """

    @classmethod
    def get_queue_kwargs(cls) -> Dict[str, Callable]:
        return {
            "loader": cls.loads,
            "dumper": cls.dumps,
        }

    @classmethod
    @abstractmethod
    def loads(cls, b: memoryview) -> Self:
        """"""

    @abstractmethod
    def dumps(self: Self) -> bytes:
        """"""
