# -*- coding: utf-8 -*-
from __future__ import annotations

from functools import reduce
from typing import ClassVar, Dict, Generator, List, Optional, Tuple

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
    """This node has only forcing children generated so far."""

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
    def score(self, value: float32, parallel_propagation: bool = True) -> None:
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
                parent.propagate_score(value, old_value)

    def propagate_parallel(self, value: float32, old_value: float32):
        """
        The idea was to somehow vectorize and run with GPU, but so far it's not yet so.
        """

        parents_list: List[Tuple[Node, List[Node]]] = list(self.generate_parents())

        for node, children in parents_list:
            if node.color:
                if node.only_forcing:
                    if value < node.score:
                        self.set_score(value)
                    else:
                        break
                else:
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
                if node.only_forcing:
                    if value < node.score:
                        self.set_score(value)
                    else:
                        break
                else:
                    if value < node.score:
                        # score increased (relatively), propagate immediately
                        self.set_score(value)
                    elif old_value is not None and value > old_value > node.score:
                        # score decreased (relatively), but old score indicates an insignificant node
                        break
                    else:
                        # score decreased, propagate the lowest child score
                        self.set_score(reduce(self.minimal, children)._score)

    def generate_parents(self) -> Generator[Tuple[Node, List[Node]], None, None]:
        """"""

        node = self.parent
        while node:
            yield node, node.children
            node = node.parent

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

        if self.parent:
            if self.leaf_level > self.parent.leaf_level:
                self.parent.leaf_level = self.leaf_level
                self.parent.leaf_color = self.leaf_color

            if not self.being_processed and self.parent.being_processed:
                # switch parent to not being processed if node is not being processed anymore
                self.parent.being_processed = False

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

        elif self.parent is not None:
            parent_score: float32 = self.parent.score

            if self.only_forcing:
                self.propagate_only_forcing(value)
            else:
                self.propagate_optimal(children, parent_score, value, old_value)

    def propagate_only_forcing(self, value: float32) -> None:
        """
        Propagate score to this node knowing that all children are forcing moves.
        In such case the value can only go one way (forcing moves are assumed to be "good" by definition).
        """

        if not self.color:
            if value < self.score:
                self.score = value
        else:
            if value > self.score:
                self.score = value

    def propagate_optimal(self, children: List[Node], parent_score: float32, value: float32, old_value: float32):
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

    def set_score(self, value: float32) -> None:
        """Set score skipping property setter (propagation)."""

        self._score = value

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
