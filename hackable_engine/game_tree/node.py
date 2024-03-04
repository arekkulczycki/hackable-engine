# -*- coding: utf-8 -*-
# pylint: disable=protected-access  #-> complains about protected access to _score, but accessed within the same class
from __future__ import annotations

from functools import reduce
from typing import ClassVar, Generator, List, Optional, Tuple

from numpy import float32

from hackable_engine.common.constants import ROOT_NODE_NAME


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
    """This node has only forcing children generated so far."""

    board: bytes
    """Serialized board representation."""

    children: List[Node]
    leaf_color: bool
    leaf_level: int

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

        self.propagate_being_processed_up()

        # assign last because requires other attributes initiated
        self.score = score

    def __repr__(self):
        return (
            f"Node({self.level}, {self.name}, {self.move}, {round(self._score, 3)}, "
            f"ini: {round(self.init_score, 3)}, {self.being_processed})"
        )

    @property
    def name(self) -> str:
        if self.parent is None:
            return ROOT_NODE_NAME

        name: str = self.move
        parent: Optional[Node] = self.parent
        while parent:
            name = (
                f"{parent.move}.{name}" if parent.parent else f"{ROOT_NODE_NAME}.{name}"
            )
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
    def score(self, value: float32, parallel_propagation: bool = True) -> None:
        """
        Set score and propagate it up the tree.

        Parallel propagation is an idea in progress, does nothing right now.
        """

        old_value: Optional[float32] = getattr(
            self, "_score", None
        )  # None means is a leaf
        self._score = value

        parent: Optional[Node] = self.parent
        if parent:
            # TODO: to not propagate is good idea only if we wait all captures to finish which is not the case
            if parallel_propagation:
                self.propagate_parallel(value, old_value)
            else:
                parent.inherit_score(value, old_value)

    def inherit_score(self, value: float32, old_value: Optional[float32]) -> None:
        """Assign the score to self from a given value or children."""

        if self.being_processed:
            # if being processed it means not all children have finished processing, therefore can wait
            pass

        elif self.parent is not None:
            parent_score: float32 = self.parent.score

            if self.only_forcing:
                self.propagate_only_forcing(value)
            else:
                self.propagate_optimal(self.children, parent_score, value, old_value)

    def propagate_only_forcing(self, value: float32) -> None:
        """
        Propagate score to this node knowing that all children are forcing moves.

        In such case the value can only go one way (forcing moves are assumed to be "good" by definition).
        If for whatever reason the score does not improve we keep the old one and skip propagation.
        """
        # pylint: disable=consider-using-max-builtin,consider-using-min-builtin

        if self.color:
            if value > self.score:
                self.score = value
        else:
            if value < self.score:
                self.score = value

    def propagate_optimal(
        self,
        children: List[Node],
        parent_score: float32,
        value: float32,
        old_value: Optional[float32],
    ):
        """"""

        if self.color:
            if value > parent_score:
                # score increased, propagate immediately
                self.score = value
            elif old_value is not None and value < old_value < parent_score:
                # score decreased, but old score indicates an insignificant node
                pass
            else:
                self.score = reduce(self.maximal, children)._score
        else:
            if value < parent_score:
                # score increased (relatively), propagate immediately
                self.score = value
            elif old_value is not None and value > old_value > parent_score:
                # score decreased (relatively), but old score indicates an insignificant node
                pass
            else:
                self.score = reduce(self.minimal, children)._score

    def propagate_parallel(self, value: float32, old_value: Optional[float32]):
        """
        The idea was to somehow vectorize and run with GPU, but so far it's not yet so.

        Work in progress...
        """

        parents_list: List[Tuple[Node, List[Node]]] = list(self.generate_parents())

        for node, children in parents_list:
            if node.only_forcing:
                # @see `propagate_only_forcing`
                self.set_score(value)

            elif node.color:
                if value > node.score:
                    # score increased, propagate immediately
                    self.set_score(value)
                elif old_value is not None and value < old_value < node.score:
                    # score decreased, but old score indicates an insignificant node
                    break
                else:
                    # score decreased, propagate the highest child score
                    self.set_score(reduce(self.maximal, children)._score)
            else:
                if value < node.score:
                    # score increased (relatively, for black), propagate immediately
                    self.set_score(value)
                elif old_value is not None and value > old_value > node.score:
                    # score decreased (relatively, for black), but old score indicates an insignificant node
                    break
                else:
                    # score decreased, propagate the lowest child score
                    self.set_score(reduce(self.minimal, children)._score)

    def generate_parents(self) -> Generator[Tuple[Node, List[Node]], None, None]:
        node = self.parent
        while node:
            yield node, node.children
            node = node.parent

    def propagate_being_processed_up(self) -> None:
        """Propagate the node feature up the tree to keep track of which branches are free to be processed."""

        if self.parent:
            if self.leaf_level > self.parent.leaf_level:
                self.parent.leaf_level = self.leaf_level
                self.parent.leaf_color = self.leaf_color

            if not self.being_processed and self.parent.being_processed:
                # switch parent to not being processed if node is not being processed anymore
                self.parent.being_processed = False

            self.parent.propagate_being_processed_up()

    def propagate_being_processed_down(self) -> None:
        """Propagate recursively down the tree simply to switch the value in the entire tree."""

        self.being_processed = False
        for child in self.children:
            child.propagate_being_processed_down()

    def set_score(self, value: float32) -> None:
        """Set score skipping property setter for performance."""

        self._score = value

    @staticmethod
    def minimal(x: Node, y: Node) -> Node:
        return x if x._score < y._score else y

    @staticmethod
    def maximal(x: Node, y: Node) -> Node:
        return x if x._score > y._score else y

    def is_descendant_of(self, node: Node) -> bool:
        """Tell if `self` is a descendant of a given node,"""

        expected = self.parent
        while expected:
            if expected is node:
                return True
            expected = expected.parent
        return False
