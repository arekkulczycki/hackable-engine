# -*- coding: utf-8 -*-
from unittest import TestCase

from numpy import float32

from arek_chess.controller import Controller
from arek_chess.game_tree.node import Node


class TestController(TestCase):
    """
    Class_docstring
    """

    def test_remap_nodes_dict(self) -> None:
        """"""

        nodes_dict = {
            "1": "a",
            "1.a": "aa",
            "1.b": "ab",
            "1.c": "ac",
            "1.a.a": "aaa",
            "1.a.b": "aab",
            "1.b.a": "aba",
            "1.b.b": "abb",
            "1.b.b.a": "abba",
            "1.b.c": "abc",
            "1.c.b": "acb",
        }

        expected_remapped_dict = {
            "1": "ab",
            "1.a": "aba",
            "1.b": "abb",
            "1.b.a": "abba",
            "1.c": "abc",
        }

        node = Node(None, "b", float32(1.0), 0, False, False, False, b"")
        remapped_dict = Controller._remap_nodes_dict(
            nodes_dict, node
        )
        assert remapped_dict == expected_remapped_dict

        ### grand ###
        expected_remapped_dict_grand = {
            "1": "abb",
            "1.a": "abba",
        }

        remapped_dict = Controller._remap_nodes_dict(
            nodes_dict, Node(node, "b", float32(1.0), 0, False, False, False, b""), grand=True
        )
        print(remapped_dict)
        assert remapped_dict == expected_remapped_dict_grand
