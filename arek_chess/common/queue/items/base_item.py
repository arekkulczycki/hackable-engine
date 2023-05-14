# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC
from dataclasses import astuple
from typing import Iterator, Tuple


# BaseItemT = TypeVar("BaseItemT")


class BaseItem(ABC):
    """
    Item passed through DispatcherQueue.
    """

    # run_id: str
    SEPARATOR: bytes = b"@@@@@"

    def __iter__(self) -> Iterator:
        return iter(astuple(self))

    def as_tuple(self) -> Tuple:
        return astuple(self)  #[1:]  # skipping run_id

    # @staticmethod
    # @abstractmethod
    # def loads(b: memoryview) -> BaseItemT:
    #     """"""
    #
    # @staticmethod
    # @abstractmethod
    # def dumps(obj: BaseItemT) -> bytes:
    #     """"""
