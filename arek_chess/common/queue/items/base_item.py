# -*- coding: utf-8 -*-

from dataclasses import astuple, dataclass


@dataclass
class BaseItem:
    """
    Item passed through DispatcherQueue.
    """

    run_id: str

    def __iter__(self):
        return iter(astuple(self))
