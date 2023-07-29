# -*- coding: utf-8 -*-
from itertools import cycle
from typing import Dict, Generator, List, Optional

from chess import BISHOP, KNIGHT
from numpy import abs, float32

from arek_chess.common.constants import INF
from arek_chess.common.exceptions import SearchFailed
from arek_chess.common.queue.items.selector_item import SelectorItem
from arek_chess.criteria.selection.fast_selector import FastSelector
from arek_chess.game_tree.node import Node

# level_to_block: cycle = cycle([3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0])
# level_to_block: cycle = cycle([4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0])
# level_to_block: cycle = cycle([7, 6, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0])
# level_to_block: cycle = cycle([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
level_to_block: cycle = cycle([12])


class Traverser:
    """
    Tree traversal model.
    """

    root: Node

    def __init__(
        self,
        root: Node,
        nodes_dict: Dict,  # WeakValueDictionary,
        transposition_dict: Optional[Dict] = None,  # WeakValueDictionary,
    ) -> None:
        super().__init__()

        self.root = root
        self.nodes_dict: Dict[str, Node] = nodes_dict
        self.transposition_dict: Optional[Dict[bytes, Node]] = transposition_dict

        # self.selector: LinearProbabilitySelector = LinearProbabilitySelector()
        # self.selector: ExpProbabilitySelector = ExpProbabilitySelector()
        """gives a chance to unlikely sequences at a cost of looking at tons of nonsense"""
        self.selector: FastSelector = FastSelector()
        """straight to the point but is more likely to miss on something"""
        # self.selector: Cluster1dSelector = Cluster1dSelector()

    def get_nodes_to_look_at(self, iterations: int = 1) -> List[Node]:
        """
        Get N nodes, taking each from a different node, but focusing around the best nodes.
        """

        maybe_nodes: Generator[Optional[Node], None, None] = (
            self.get_next_node_to_look_at() for _ in range(iterations)
        )
        nodes: List[Node] = [node for node in maybe_nodes if node is not None]

        return nodes

    def get_next_node_to_look_at(self) -> Optional[Node]:
        """"""

        # TODO: cache best node (level>1) and start from it instead of starting from scratch
        #  define interesting nodes and then cache them

        best_node: Node = self.root
        children: List[Node]

        k = 0
        while True:
            children = best_node.children
            if not children or best_node.only_captures:
                if best_node is self.root:
                    # haven't received child nodes evaluations yet
                    return None

                if abs(best_node.score) == INF:
                    # the best path leads to checkmate then don't select anything more
                    return None

                # is a leaf that hasn't yet been fully looked at (could have recaptures looked at)
                best_node.being_processed = True
                return best_node

            level = best_node.level
            free_children = [
                node
                for node in children
                if not node.being_processed and node.level > level
            ]

            if free_children:
                best_node = self.select_promising_node(free_children, best_node.color)
            else:
                if best_node.parent is None:
                    return None

                # all children being processed, go up the tree again
                best_node.being_processed = True
                best_node = best_node.parent

                continue

    def select_promising_node(self, nodes: List[Node], color: bool) -> Node:
        """
        Get the child node, selected based on implemented criteria.
        """

        return self.selector.select(nodes, color)

    def create_nodes_and_autodistribute(
        self, candidates: List[SelectorItem]
    ) -> List[Node]:
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
                node = (
                    self.transposition_dict.get(candidate.board)
                    if self.transposition_dict is not None
                    else None
                )
            except KeyError as e:
                raise SearchFailed(
                    f"node not found in items: {candidate.node_name}, "
                    f"{candidate.run_id}, {candidate.move_str}, {self.root.move}"
                ) from e

            # should_search_recaptures: bool = self._is_good_capture_in_top_branch(parent, candidate.captured, finished)
            # should_search_recaptures: bool = candidate.captured > 0 and (
            #     not finished or self._is_good_recapture(parent, candidate.captured, candidate.score)
            # )

            if parent.level < 1:
                should_search = True
            else:
                should_search = candidate.captured > 0 and (
                    (self.root.color and bool(candidate.score > self.root.score))
                    or (not self.root.color and bool(self.root.score > candidate.score))
                )

            if node is not None:
                # node already existed from different parent
                name = ".".join((candidate.node_name, candidate.move_str))
                self.nodes_dict[name] = node

                parent.children.append(node)
                parent.propagate_score(node.score, None, node.color, node.level)
                parent.being_processed = False

            else:
                node = self.create_node(
                    parent,
                    candidate.move_str,
                    candidate.score,
                    candidate.captured,
                    not parent.color,
                    bool(should_search),
                    candidate.board,
                )

                # analyse "good" captures immediately if they are in the top branch
                if node and should_search:
                    nodes_to_distribute.append(node)

        return nodes_to_distribute

    def _is_good_recapture(self, parent: Node, captured: int, score: float32) -> bool:
        """"""

        if (
            captured > parent.captured
            and not (
                captured == BISHOP and parent.captured == KNIGHT  # not bishop vs knight
            )
            and (
                (score > self.root.score and self.root.color)
                or (score < self.root.score and not self.root.color)
            )
        ):
            return True

        return False

    def create_node(
        self,
        parent: Node,
        move: str,
        score: float32,
        captured: int,
        color: bool,
        should_search: bool,
        board: bytes,
    ) -> Optional[Node]:
        """"""

        node: Node = Node(
            parent=parent,
            move=move,
            score=score,
            captured=captured,
            color=color,
            being_processed=should_search,
            only_captures=should_search and bool(captured),
            board=board,
        )

        self.nodes_dict[node.name] = node
        if self.transposition_dict is not None:
            self.transposition_dict[board] = node

        return node

    def get_node(self, node_name: str) -> Node:
        """"""

        return self.nodes_dict[node_name]
