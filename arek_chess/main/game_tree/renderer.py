"""
Tree renderer class.
"""

from typing import Optional, Generator

from anytree import Node, RenderTree
from anytree.render import _is_last

from arek_chess.constants import INF


class PrunedTreeRenderer(RenderTree):
    """"""

    def __init__(self, root: Node, *, depth: int = 0, path: str = "", **kwargs) -> None:
        """"""

        super().__init__(root, **kwargs)

        self.depth = depth
        self.path = path

    def __iter__(self) -> Generator:
        """"""

        return self.__next(self.node, tuple())

    def __next(self, node: Node, continues, level=0) -> Generator:
        """"""

        yield self._RenderTree__item(node, continues, self.style)
        children = node.children
        new_children = ()
        for i in range(len(children)):
            child = children[i]
            if self.path and (level > 0) and (self.path not in node.name):
                continue
            if self.has_deep_family(child):  # and self.path in node.name:
                new_children += (child,)
        children = new_children

        level += 1
        if children and (self.maxlevel is None or level < self.maxlevel):
            children = self.childiter(children)
            for child, is_last in _is_last(children):
                for grandchild in self.__next(
                    child, continues + (not is_last,), level=level
                ):
                    yield grandchild

    def has_deep_family(self, node: Node) -> bool:
        """"""

        if node.level >= self.depth or node.score in [INF, -INF]:
            return True

        for i in node.children:
            if self.has_deep_family(i):
                return True

        return False
