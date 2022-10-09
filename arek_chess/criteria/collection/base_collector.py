from abc import ABC
from typing import Dict, List, Union


class BaseCollector(ABC):
    """
    Inherit from this class to implement your own collector.

    Provides ordering of candidate moves in order to save time on analysing less hopeful branches.

    Must implement just the order method
    """

    def order(
        self, candidates: List[Dict[str, Union[str, int, float]]], color: bool
    ) -> List[Dict[str, Union[str, int, float]]]:
        """

        :param candidates: list of all legal moves in the position
        :param color: color of the player to choose from above candidates

        :return: sorted (or also filtered) list of moves that was received as first argument
        """

        raise NotImplementedError
