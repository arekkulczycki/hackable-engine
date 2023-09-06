# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import reduce
from typing import ClassVar, Dict, List, Optional

from numpy import float32

from arek_chess.common.constants import ROOT_NODE_NAME


class Node:
    """
    Node that propagates fast through its tree.
    """

    _score: float32
    init_score: float32

    parent: Optional[Node]
    move: str
    forcing_level: int
    color: bool
    being_processed: bool
    only_forcing: bool
    board: bytes

    children: List[Node]
    leaf_color: bool
    leaf_level: int

    # debug_log: ClassVar[Dict] = {}
    debug_log: ClassVar[List] = []
    total: int = 0

    def __init__(
        self,
        parent: Optional[Node],
        move: str,
        score: float32,
        forcing_level: int,
        color: bool,
        board: bytes,
    ):
        self.parent = parent
        if parent:
            parent.children.append(self)

        self.move = move

        self.init_score = score
        self.forcing_level = forcing_level
        self.color = color
        self.being_processed = False
        """Indicates that there is no descendant to be processed - all are currently being processed."""

        self.only_forcing = False
        self.board = board

        self.children = []
        self.leaf_color = color
        self.leaf_level = self.level

        # if parent:
        #     self.debug_log.append(
        #         ("create", self.name, "\n")
        #     )
        self.propagate_being_processed_up()

        self.score = score
        """assign last because requires other attributes initiated"""

    def __repr__(self):
        return f"Node({self.level}, {self.name}, {self.move}, {round(self._score, 3)}, ini: {round(self.init_score, 3)}, {self.being_processed})"

    @property
    def name(self) -> str:
        if self.parent is None:
            return ROOT_NODE_NAME

        name: str = self.move
        parent: Optional[Node] = self.parent
        while parent:
            name = f"{parent.move}.{name}" if parent.parent else f"{ROOT_NODE_NAME}.{name}"
            parent = parent.parent

        return name

    @property
    def level(self) -> int:
        level = 0

        parent = self.parent
        while parent:
            level += 1
            parent = parent.parent

        return level

    @property
    def score(self) -> float32:
        return self._score

    @score.setter
    def score(self, value: float32) -> None:
        old_value: Optional[float32] = getattr(
            self, "_score", None
        )  # None means is a leaf
        self._score = value

        parent: Optional[Node] = self.parent
        if parent:
            # switch parent to the same status if is processing only forcing moves (edit: but why so?)
            # parent.being_processed = self.being_processed if self.only_forcing else False

            # if not self.being_processed:  # don't propagate until capture-fest finished
            # TODO: to not propagate is good idea only if we wait all captures to finish which is not the case
            parent.propagate_score(value, old_value)

    # @property
    # def being_processed(self) -> bool:
    #     return self._being_processed
    #
    # @being_processed.setter
    # def being_processed(self, v: bool) -> None:
    #     self.debug_log.append(
    #         ("process", self.name, str(v), "\n")
    #     )
    #     self._being_processed = v

    def propagate_being_processed_up(self) -> None:
        """"""

        if self.parent and self.parent.being_processed:
            # switch parent to not being processed if node is not being processed anymore
            # if self.only_forcing and self.being_processed:
            #     self.parent.being_processed = True
            # else:
            if not self.being_processed:
                self.parent.being_processed = False
                if self.leaf_level > self.parent.leaf_level:
                    self.parent.leaf_level = self.leaf_level
                    self.parent.leaf_color = self.leaf_color
                self.parent.propagate_being_processed_up()

    def propagate_being_processed_down(self) -> None:
        """"""

        self.being_processed = False
        for child in self.children:
            child.propagate_being_processed_down()

    def propagate_score(
        self, value: float32, old_value: Optional[float32]
    ) -> None:
        """"""

        score: Optional[float32]
        propagating_children: List[Node]

        children = self.children

        if not children:
            self.set_score(value)

        elif self.being_processed:
            # if being processed it means not all forcing children have finished processing, therefore can wait
            pass

        elif self.only_forcing and self.parent is not None:
            if not self.color:
                recapture_score = reduce(self.minimal, children)._score
                if recapture_score >= self.init_score:
                    # if opponent doesn't have any good recaptures then keeping the score
                    # score = self.init_score
                    pass
                else:
                    # not being processed anymore, final propagation,
                    # not taking the recapture score as may have non-capture children
                    score = max(self.parent.init_score, recapture_score)
                    self.set_score(score)
            else:
                recapture_score = reduce(self.maximal, children)._score
                if recapture_score <= self.init_score:
                    # if opponent doesn't have any good recaptures then keeping the score
                    # score = self.init_score
                    pass
                else:
                    # final propagation, not taking the recapture score as may have non-capture children
                    score = min(self.parent.init_score, recapture_score)
                    self.set_score(score)

        else:
            parent_score: float32 = self._score
            if self.color:
                if value > parent_score:
                    # score increased, propagate immediately
                    self.set_score(value)
                elif old_value is not None and value < old_value < parent_score:
                    # score decreased, but old score indicates an insignificant node
                    pass
                else:
                    self.set_score(reduce(self.maximal, children)._score)
            else:
                if value < parent_score:
                    # score increased (relatively), propagate immediately
                    self.set_score(value)
                elif old_value is not None and value > old_value > parent_score:
                    # score decreased (relatively), but old score indicates an insignificant node
                    pass
                else:
                    self.set_score(reduce(self.minimal, children)._score)

    def set_score(self, value: float32) -> None:
        self.score = value
        # TODO: set `being_processed=False` here and remove propagation on init?
        #  (currently doesn't even get here if is `being_processed`)

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
