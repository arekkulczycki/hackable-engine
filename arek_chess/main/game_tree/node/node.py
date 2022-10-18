# -*- coding: utf-8 -*-
"""
Node that propagates fast through its tree.
"""

from __future__ import annotations

from functools import reduce
from typing import Optional, List

from numpy import double


class Node:
    """
    Node that propagates fast through its tree.
    """

    _score: double

    def __init__(
        self,
        parent: Optional[Node],
        name: str,
        move: str,
        score: double,
        captured: int,
        level: int,
        color: bool,
    ):
        self.parent: Optional[Node] = parent
        if parent:
            parent.children.append(self)

        self.name: str = name
        self.move: str = move
        self.score: double = score
        self.init_score: double = score
        self.captured: int = captured
        self.level: int = level
        self.color: bool = color

        self.children: List[Node] = []
        self.looked_at: bool = False
        self.being_processed: bool = False

    def __repr__(self):
        return f"Node({self.level}, {self.name}, {round(self._score, 3)}, initial: {round(self.init_score, 3)})"

    @property
    def score(self) -> double:
        return self._score

    @score.setter
    def score(self, value: double) -> None:
        old_value: Optional[double] = getattr(self, "_score", None)
        self._score = value

        parent: Optional[Node] = self.parent
        if parent:
            parent.propagate_score(value, old_value)

    def propagate_score(self, value: double, old_value: Optional[double]) -> None:
        """"""

        children: List[Node] = self.children

        if not children:
            self.score = value

        else:
            parent_score: double = self._score
            if self.color:
                if value > parent_score:
                    self.score = value
                elif value < parent_score and (old_value is None or parent_score <= old_value):
                    self.score = reduce(self.maximal, children)._score
            else:
                if value < parent_score:
                    self.score = value
                elif value > parent_score and (old_value is None or parent_score >= old_value):
                    self.score = reduce(self.minimal, children)._score

    @staticmethod
    def minimal(x: Node, y: Node) -> Node:
        return x if x._score < y._score else y

    @staticmethod
    def maximal(x: Node, y: Node) -> Node:
        return x if x._score > y._score else y
