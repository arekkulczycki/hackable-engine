# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Optional, Dict

from hackable_engine.game_tree.node import Node


@dataclass(slots=True)
class SearchTreeCache:
    """
    Holds data during and after a finished tree search.
    """

    root: Optional[Node]
    nodes_dict: Dict[str, Node]  # WeakValueDictionary
    transposition_dict: Optional[Dict[bytes, Node]] = None
    """If None then transpositions will not be used in the search process."""


@dataclass(slots=True)
class SideTreeCache:
    """
    Holds the tree cache for two independent analysis runs, for instance if two engines play each other.

    Provides logic associated with use cases of the cache.
    """

    white: SearchTreeCache
    """A tree cache seen from white player perspective."""

    black: SearchTreeCache
    """A tree cache seen from black player perspective."""

    @staticmethod
    def get_tree_branch(search_tree: SearchTreeCache, move_uci: str) -> SearchTreeCache:
        """Get a branch from the current search tree, depending on the move that has been played."""

        search_tree.root = SideTreeCache._get_next_root(search_tree, move_uci)
        search_tree.nodes_dict = SideTreeCache._remap_nodes_dict(
            search_tree.nodes_dict, search_tree.root
        )
        # search_tree.transposition_dict = self._remap_transposition_dict  # TODO: figure out

        if search_tree.root:  # TODO: remove when works as expected
            print(
                f"reused node by {'white' if search_tree.root.color else 'black'}: "
                f"{search_tree.root.name}, depth: {search_tree.root.leaf_level - 1}"
            )

        return search_tree

    def get_color_based_tree_branch(
        self,
        search_tree: SearchTreeCache,
        color: bool,
        child_move: str,
        grandchild_move: str,
    ) -> SearchTreeCache:
        """Get a branch from the current search tree, stepping down the tree twice to get the same color root."""

        old_root = self.black.root if color else self.white.root
        old_nodes_dict = self.black.nodes_dict if color else self.white.nodes_dict
        search_tree.root = SideTreeCache._get_next_grandroot(
            old_root, child_move, grandchild_move
        )
        if search_tree.root and search_tree.root.children:
            search_tree.nodes_dict = SideTreeCache._remap_nodes_dict(
                old_nodes_dict, search_tree.root, grand=True
            )
            # next_transposition_dict = self.last_black_transposition_dict
        else:
            search_tree.nodes_dict = {}

        return search_tree

    @staticmethod
    def _get_next_root(search_tree: SearchTreeCache, move_uci: str) -> Node:
        """Return a child node of the last root that corresponds with a given move."""

        chosen_child: Optional[Node] = None
        for child in search_tree.root.children:
            if child.move == move_uci:
                chosen_child = child
                break

        if chosen_child is None:
            raise ValueError(
                f"Next root: Could not recognize move played: {move_uci}. "
                f"Children: {' - '.join([str(child) for child in search_tree.root.children])}"
            )

        return chosen_child

    @staticmethod
    def _get_next_grandroot(
        root_node: Node, child_move_uci: str, grandchild_move_uci: str
    ) -> Optional[Node]:
        """Return a grandchild node of the last root that corresponds with two last played moves."""

        chosen_child: Optional[Node] = None
        for child in root_node.children:
            if child.move == child_move_uci:
                chosen_child = child
                break

        if chosen_child is None:
            raise ValueError(
                f"Next grand-root: Could not recognize move played: {child_move_uci}. "
                f"Children: {' - '.join([str(child) for child in root_node.children])}"
            )

        chosen_grandchild: Optional[Node] = None
        for child in chosen_child.children:
            if child.move == grandchild_move_uci:
                chosen_grandchild = child
                break

        return chosen_grandchild

    @staticmethod
    def _remap_nodes_dict(
        nodes_dict: Dict, next_root: Node, grand: bool = False
    ) -> Dict:
        """
        Clean hashmap of discarded moves and rename remaining keys.
        """

        cut = 3 if grand else 2

        new_nodes_dict = {"1": next_root}
        for key, value in nodes_dict.items():
            key_split = key.split(".")
            split_len = len(key_split)

            if grand and (
                split_len < 4  # discard first and second level children
                # discard all nodes that are not descendants of `next_root`
                or key_split[1] != next_root.parent.move
                or key_split[2] != next_root.move
            ):
                continue
            # if not `grand` then discard root and all nodes that are not descendants of `next_root`
            if not grand and (split_len < 3 or key_split[1] != next_root.move):
                continue

            new_key = ".".join(["1"] + key_split[cut:])
            new_nodes_dict[new_key] = value

        return new_nodes_dict
