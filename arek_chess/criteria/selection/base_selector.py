# -*- coding: utf-8 -*-

from abc import ABC
from typing import List

from arek_chess.game_tree.node import Node


class BaseSelector(ABC):
    """
    Inherit from this class to implement your own selector.

    Provides a way to randomize or artificially broaden choice of which branches to look at.

    Must implement just the select method.
    """

    @staticmethod
    def select(nodes: List[Node], color: bool) -> Node:
        """

        :param nodes: list of nodes to be selected from  # TODO: in future probably generator instead of list
        :param color: color for which the scores are decided (white needs higher scores, black - lower)

        :returns: return index to be selected
        """

        raise NotImplementedError
