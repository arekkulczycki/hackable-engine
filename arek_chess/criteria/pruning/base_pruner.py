# -*- coding: utf-8 -*-

from abc import ABC
from typing import Dict

from anytree import Node


class BasePruner(ABC):
    """
    Inherit from this class to implement your own pruner.

    Provides decision if branches from the given node should be discarded.

    Must implement just the should_prune method.
    """

    def should_prune(
        self,
        tree_stats: Dict[int, Dict[str, float]],
        score: float,
        parent_node: Node,
        color: bool,
        captured: int,
        depth: int,
    ) -> bool:
        """
        Decide which branches to prune in order to narrow down the search tree and allow deeper analysis.

        :param tree_stats: statistics of the entire up-to-this-point analysed tree
            first key is the level of the tree, 0 being the root node
            inner keys: number, min, max, mean, median (optional when number > 3), stdev (optional when number > 3)
        :param score: score of the move done from position of the given node
        :param parent_node: children of this node will become leaves if a True is returned from this method
            available attributes: level, score, children (list of Nodes), parent (Node), move (uci str)
        :param color: color of the player making the move from this node
        :param captured: flag of the captured piece, if the score reflects a move that was a capture
            0 - no capture, 1 - PAWN, 2 - KNIGHT, 3 - BISHOP, 4 - ROOK, 5 - QUEEN
        :param depth: default depth at which the search should stop

        :return: decision if branches from this node should be pruned
        """

        raise NotImplementedError
