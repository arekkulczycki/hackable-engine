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

        new_root = Node(None, "b", float32(1.0), 0, False, False, False, b"")

        nodes_dict = {
            "1": "a",
            "1.a": "aa",
            "1.b": new_root,
            "1.c": "ac",
            "1.a.aa": "aaa",
            "1.a.ab": "aab",
            "1.b.ba": "aba",
            "1.b.bb": "abb",
            "1.b.bb.bba": "abba",
            "1.b.bc": "abc",
            "1.c.cb": "acb",
        }

        expected_remapped_dict = {
            "1": new_root,
            "1.ba": "aba",
            "1.bb": "abb",
            "1.bb.bba": "abba",
            "1.bc": "abc",
        }

        remapped_dict = Controller._remap_nodes_dict(
            nodes_dict, new_root
        )
        assert remapped_dict == expected_remapped_dict

        ### grand ###
        new_grand_root = Node(new_root, "bb", float32(1.0), 0, False, False, False, b"")
        expected_remapped_dict_grand = {
            "1": new_grand_root,
            "1.bba": "abba",
        }

        remapped_dict = Controller._remap_nodes_dict(
            nodes_dict, new_grand_root, grand=True
        )

        assert remapped_dict == expected_remapped_dict_grand

    def test_remap_nodes_dict_persist_parenthood(self) -> None:
        """"""

        a = Node(None, "a", float32(1.0), 0, False, False, False, b"")
        aa = Node(a, "aa", float32(1.0), 0, False, False, False, b"")
        aaa = Node(aa, "aaa", float32(1.0), 0, False, False, False, b"")
        aab = Node(aa, "aab", float32(1.0), 0, False, False, False, b"")
        ab = Node(a, "ab", float32(1.0), 0, False, False, False, b"")
        aba = Node(ab, "aba", float32(1.0), 0, False, False, False, b"")
        abb = Node(ab, "abb", float32(1.0), 0, False, False, False, b"")
        abba = Node(abb, "abba", float32(1.0), 0, False, False, False, b"")
        abc = Node(ab, "abc", float32(1.0), 0, False, False, False, b"")
        ac = Node(a, "ac", float32(1.0), 0, False, False, False, b"")
        acb = Node(ac, "acb", float32(1.0), 0, False, False, False, b"")
        nodes_dict = {
            "1": a,
            "1.aa": aa,
            "1.ab": ab,
            "1.ac": ac,
            "1.aa.aaa": aaa,
            "1.aa.aab": aab,
            "1.ab.aba": aba,
            "1.ab.abb": abb,
            "1.ab.abb.abba": abba,
            "1.ab.abc": abc,
            "1.ac.acb": acb,
        }

        ### grand ###
        expected_remapped_dict_grand = {
            "1": abb,
            "1.abba": abba,
        }

        remapped_dict = Controller._remap_nodes_dict(
            nodes_dict, abb, grand=True
        )

        assert remapped_dict == expected_remapped_dict_grand
        assert abba.parent is abb
