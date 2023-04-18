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
    init_score: double

    parent: Optional[Node]
    name: str
    move: str
    captured: int
    level: int
    color: bool
    being_processed: bool

    children: List[Node]
    looked_at: bool
    leaf_color: bool

    def __init__(
        self,
        parent: Optional[Node],
        name: str,
        move: str,
        score: double,
        captured: int,
        level: int,
        color: bool,
        being_processed: bool,
    ):
        self.parent = parent
        if parent:
            parent.children.append(self)

        self.name = name
        self.move = move

        self.init_score = score
        self.captured = captured
        self.level = level
        self.color = color
        self.being_processed = being_processed

        self.children = []
        self.looked_at = False
        self.leaf_color = color

        self.score = score
        """assign last because requires other attributes initiated"""

    def __repr__(self):
        return f"Node({self.level}, {self.name}, {round(self._score, 3)}, initial: {round(self.init_score, 3)}, {self.being_processed})"

    @property
    def score(self) -> double:
        return self._score

    @score.setter
    def score(self, value: double) -> None:
        old_value: Optional[double] = getattr(
            self, "_score", None
        )  # None means is a leaf
        self._score = value

        parent: Optional[Node] = self.parent
        if parent:
            # once is no longer processed also propagate it
            # is important to not reset too early to prevent digging into branch that has a capture-fest analysed
            # TODO: this potentially will leave some nodes processed forever, harmful or not?
            # TODO: should do the children iteration just once? (including the one below)
            # parent.being_processed = any(child.being_processed for child in parent.children)
            parent.being_processed = False

            if (
                not self.being_processed
            ):  # and not (old_value is None and odd_level):  # odd level leafs don't propagate
                parent.propagate_score(value, old_value, self.leaf_color)

    def propagate_score(
        self, value: double, old_value: Optional[double], leaf_color: bool
    ) -> None:
        """"""

        propagating_children: List[Node]

        children = self.children

        if not children:
            self.set_score(value, leaf_color)

        else:
            parent_score: double = self._score
            if self.color:
                if value > parent_score:
                    # score increased, propagate immediately
                    self.set_score(value, leaf_color)
                elif old_value is not None and value < old_value < parent_score:
                    # score decreased, but old score indicates an insignificant node
                    pass
                else:
                    self.set_score(reduce(self.maximal, children)._score, leaf_color)
            else:
                if value < parent_score:
                    # score increased (relatively), propagate immediately
                    self.set_score(value, leaf_color)
                elif old_value is not None and value > old_value > parent_score:
                    # score decreased (relatively), but old score indicates an insignificant node
                    pass
                else:
                    self.set_score(reduce(self.minimal, children)._score, leaf_color)

    def set_score(self, value: double, leaf_color: bool) -> None:
        self.score = value
        self.leaf_color = leaf_color

    def has_grand_children(self) -> bool:
        for child in self.children:
            if child.children:
                return True
        return False

    @staticmethod
    def minimal(x: Node, y: Node) -> Node:
        return x if x._score < y._score else y

    @staticmethod
    def maximal(x: Node, y: Node) -> Node:
        return x if x._score > y._score else y

    def is_descendant_of(self, node: Node) -> bool:
        expected = self.parent
        while expected:
            if expected is node:
                return True
            expected = expected.parent
        return False
