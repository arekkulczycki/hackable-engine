from typing import Self

from hackable_engine.common.queue.items.base_item import BaseItem


class PriorityItem:
    def __init__(self, item: BaseItem, root_color: bool):
        self.item = item
        self.root_color = root_color
        """
        If white, then looking for the best black move.
        This means looking for the lowest score on odd levels and highest scores on even levels.
        """
        self.chosen_children: set[str] = set()
        self.children_count: int = 0

    def add_child(self, child: str):
        self.chosen_children.add(child)
        self.children_count += 1

    def level(self, item: BaseItem):
        """Example: 1.a1 is level 1"""
        return item.node_name.count(".")

    def __lt__(self, other: Self):
        """Priority definition for asyncio.PriorityQueue"""

        if self.item.forcing_level != other.item.forcing_level:
            return self.item.forcing_level > other.item.forcing_level

        if self.children_count != other.children_count:
            return self.children_count < other.children_count

        self_level = self.level(self.item)
        other_level = self.level(other.item)

        if is_even(self_level) and is_even(other_level):
            if self.root_color:
                return self.item.score > other.item.score
            else:
                return self.item.score < other.item.score

        if not is_even(self_level) and not is_even(other_level):
            if self.root_color:
                return self.item.score < other.item.score
            else:
                return self.item.score > other.item.score

        # # remaining case is one of them being even and the other being odd
        # if self.item.score != other.item.score:
        #     if self.root_color:
        #         # looking for the best black move therefore choose the lower score
        #         return self.item.score < other.item.score
        #     else:
        #         # looking for the best white move therefore choose the higher score
        #         return self.item.score > other.item.score

        # if same score then dig into the deeper branch
        return self_level > other_level

def is_even(level: int):
    return level % 2 == 0
