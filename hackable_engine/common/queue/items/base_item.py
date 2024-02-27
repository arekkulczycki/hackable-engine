# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import astuple
from typing import Callable, Dict, Iterator, Tuple, TypeVar

BaseItemT = TypeVar("BaseItemT")


class BaseItem(ABC):
    """
    Item passed through DispatcherQueue.
    """

    def __iter__(self) -> Iterator:
        return iter(astuple(self))

    def as_tuple(self) -> Tuple:
        return astuple(self)  #[1:]  # skipping run_id

    @classmethod
    def get_queue_kwargs(cls) -> Dict[str, Callable]:
        return {
            "loader": cls.loads,
            "dumper": cls.dumps,
        }

    @staticmethod
    @abstractmethod
    def loads(b: memoryview) -> BaseItemT:
        """"""

    @staticmethod
    @abstractmethod
    def dumps(obj: BaseItemT) -> bytes:
        """"""
