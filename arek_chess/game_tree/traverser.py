# -*- coding: utf-8 -*-

from itertools import cycle
from typing import Dict, List, Optional, Generator
from weakref import WeakValueDictionary

from chess import BISHOP, KNIGHT
from numpy import abs, float32

from arek_chess.common.constants import INF
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.criteria.selection.fast_selector import FastSelector
from arek_chess.game_tree.node import Node

level_to_block: cycle = cycle([3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0])


class Traverser:
    """
    Tree traversal model.
    """

    root: Node

    def __init__(self, root: Node, nodes_dict: WeakValueDictionary) -> None:
        super().__init__()

        self.root = root
        self.nodes_dict: WeakValueDictionary = nodes_dict

        # self.selector: LinearProbabilitySelector = LinearProbabilitySelector()
        # self.selector: ExpProbabilitySelector = ExpProbabilitySelector()
        """gives a chance to unlikely sequences at a cost of looking at tons of nonsense"""
        self.selector: FastSelector = FastSelector()
        """straight to the point but is more likely to miss on something"""
        # self.selector: Cluster1dSelector = Cluster1dSelector()

        self.last_best_node: Optional[Node] = None

        self.node_to_block: Optional[Node] = None
        self.blocked_nodes: List[Node] = []

        self.selections: Dict[Node, int] = {}

    def get_nodes_to_look_at(self, iterations: int = 1) -> List[Node]:
        """
        Get N nodes, taking each from a different node, but focusing around the best nodes.
        """

        iterations = 2 * iterations
        # level_to_block: cycle = cycle(
        #     [3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0][-iterations:]
        # )
        leaf_color: Generator[bool, None, None] = (
            (i % 2 == 1) if self.root.color else (i % 2 != 1) for i in range(iterations)
        )
        maybe_nodes: Generator[Optional[Node], None, None] = (
            self._get_next_node_to_look_at_and_block(
                next(leaf_color), next(level_to_block)
            )
            for _ in range(iterations)
        )
        nodes: List[Node] = [node for node in maybe_nodes if node is not None]
        """should challenge the best branch by first picking up odd-level leaf, i.e. leaf opposite to root color"""

        # unblock temporarily blocked nodes
        for child in self.blocked_nodes:
            child.being_processed = False
        self.blocked_nodes.clear()

        return nodes

    def _get_next_node_to_look_at_and_block(
        self, leaf_color: bool, level_to_block: int = 0
    ) -> Optional[Node]:
        node: Optional[Node] = self.get_next_node_to_look_at(leaf_color, level_to_block)

        if self.node_to_block is not None and not self.node_to_block.being_processed:
            self.node_to_block.being_processed = True  # temporarily blocking
            self.blocked_nodes.append(self.node_to_block)
            self.node_to_block = None

        return node

    def get_next_node_to_look_at(
        self, leaf_color: bool, level_to_block: int = 0
    ) -> Optional[Node]:
        """"""

        # TODO: cache best node (level>1) and start from it instead of starting from scratch
        #  define interesting nodes and then cache them

        best_node: Node = self.root
        children: List[Node]

        k = 0
        while True:
            children = best_node.children
            if not children:
                if best_node is self.root:
                    # haven't received child nodes evaluations yet
                    return None

                if abs(best_node.score) == INF:
                    # the best path leads to checkmate then don't select anything more
                    return None

                # is a leaf that hasn't yet been looked at
                best_node.being_processed = True
                return best_node

            # children of right leaf color, except ones being processed or checkmate
            free_children: List[Node] = []
            free_correct_leaf_children: List[Node] = []
            for node in children:
                if not node.being_processed and node.leaf_color == leaf_color:
                    free_correct_leaf_children.append(node)
                elif not node.being_processed:
                    free_children.append(node)

            if free_correct_leaf_children:
                best_node = self.select_promising_node(
                    free_correct_leaf_children, best_node.color
                )
            elif free_children:
                best_node = self.select_promising_node(free_children, best_node.color)
            else:
                return None

            if k == level_to_block:
                self.node_to_block = best_node

                # if best_node in self.selections:
                #     self.selections[best_node] += 1
                # else:
                #     self.selections[best_node] = 1
                # print(f"top node: {best_node.name}, {best_node.score}")
            k += 1

    def select_promising_node(self, nodes: List[Node], color: bool) -> Node:
        """
        Get the child node, selected based on implemented criteria.
        """

        return self.selector.select(nodes, color)

    def get_nodes_to_distribute(self, candidates: List[SelectorItem]) -> List[Node]:
        """"""

        candidate: SelectorItem
        parent: Node
        node: Optional[Node]
        level: int
        nodes_to_distribute: List[Node] = []

        # best_score = math.inf if color else -math.inf
        for candidate in candidates:
            try:
                parent = self.get_node(candidate.node_name)
                # parent_score = parent.score
            except KeyError as e:
                if not self.root.children:
                    # getting some leftover evaluations from the previous run
                    continue

                raise SearchFailed(
                    f"node not found in items: {candidate.node_name}, "
                    f"{candidate.run_id}, {candidate.move_str}, {self.root.move}"
                ) from e

            level = candidate.node_name.count(".") + 1
            color: bool = self.root.color if level % 2 == 0 else not self.root.color

            should_search_recaptures: bool = self._is_good_capture_in_top_branch(
                parent, candidate.captured
            )

            node = self.create_node(
                parent,
                candidate.move_str,
                candidate.score,
                candidate.captured,
                color,
                should_search_recaptures,
            )

            # analyse "good" captures immediately if they are in the top branch
            if node and should_search_recaptures:
                nodes_to_distribute.append(node)

        return nodes_to_distribute

    def _is_good_capture_in_top_branch(self, parent: Node, captured: int) -> bool:
        """"""

        # return False

        # if captured > 0:
        if captured > 1:
            # check if is winning material
            if captured > parent.captured and not (
                captured == BISHOP and parent.captured == KNIGHT
            ):  # not bishop vs knight
                return True

            # if (  # check if is a "good" capture
            #     parent is None
            #     or not parent.captured  # first capture
            #     or (  # such capture exchange that score is higher than grandparent score
            #         parent.parent is not None
            #         and (
            #             (color and score < parent.parent.init_score)
            #             or (not color and score > parent.parent.init_score)
            #         )
            #     )
            # ):
            #     if parent.is_descendant_of(self.last_best_node):
            #         return True
        return False

    def create_node(
        self,
        parent: Node,
        move: str,
        score: float32,
        captured: int,
        color: bool,
        should_process: bool,
    ) -> Optional[Node]:
        """"""

        # if captured == -1:  # node reversed from distributor when found checkmate
        #     try:
        #         self.nodes_dict[child_name].score = score
        #         self.nodes_dict[child_name].captured = -1
        #         return None
        #     except KeyError:
        #         # was never meant to be here, but somehow queue delivers phantom items
        #         pass

        # print(f"creating: {move}, {parent.name}")
        node: Node = Node(
            parent=parent,
            move=move,
            score=score,
            captured=captured,
            color=color,
            being_processed=should_process,
        )

        self.nodes_dict[node.name] = node

        return node

    def get_node(self, node_name: str) -> Node:
        """"""

        return self.nodes_dict[node_name]
