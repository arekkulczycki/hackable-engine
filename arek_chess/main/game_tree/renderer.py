"""
Tree renderer class.
"""

from anytree import Node, RenderTree
from anytree.render import _is_last

from arek_chess.main.game_tree.constants import INF


class PrunedTreeRenderer(RenderTree):
    def __init__(self, depth: int, *args, **kwargs):
        self.depth = depth
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return self.__next(self.node, tuple())

    def __next(self, node, continues, level=0):
        yield self._RenderTree__item(node, continues, self.style)
        children = node.children
        new_children = ()
        for i in range(len(children)):
            child = children[i]
            if self.has_deep_family(child):
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

    def has_deep_family(self, node: Node):
        if node.level >= self.depth or node.score in [INF, -INF]:
            return True

        for i in node.children:
            if self.has_deep_family(i):
                return True

        return False
