# -*- coding: utf-8 -*-

from random import random
from unittest import TestCase

from arek_chess.criteria.selection.square_probability_selector import (
    SquareProbabilitySelector,
)


class TestSquareProbabilitySelector(TestCase):

    def test_normalize_scores(self) -> None:
        for i in range(5):
            scores = [(random() - 0.5) * 100 for _ in range(100)]

            assert all(
                [
                    0 <= score <= 1
                    for score in SquareProbabilitySelector._normalized_weights(
                        scores, bool(i)
                    )
                ]
            )
