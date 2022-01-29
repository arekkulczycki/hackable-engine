# -*- coding: utf-8 -*-
"""
Tree node class.
"""

from anytree import Node


class BackpropNode(Node):
    _score: float

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._score = kwargs.get("score", None)

    def __repr__(self):
        return f"Node({self.level}, {self.color}, {self.move}, {getattr(self, 'deep', False)}, " \
               f"{round(self._score, 3)}::{round(self.init_score)})"

    def __lt__(self, other):
        return self.score < other.score

    def __gt__(self, other):
        return self.score > other.score

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        self._score = value

        parent = self.parent
        if parent and parent.score != value:
            children = parent.children
            if self.color:
                best_score = max([child.score for child in children])
            else:
                best_score = min([child.score for child in children])

            if parent.score != best_score:
                parent.score = best_score
