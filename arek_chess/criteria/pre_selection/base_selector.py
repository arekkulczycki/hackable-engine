from abc import ABC
from typing import Dict, List, Union


class BaseSelector(ABC):
    """
    Inherit from this class to implement your own pre-selector.

    Provides a pre-selection of candidate moves logic in order to narrow the search even before going in depth.

    Must implement just the select method
    """

    def select(
        self, candidates: Dict[str, Union[str, int, float]], color: bool
    ) -> List[Dict[str, Union[str, int, float]]]:
        """

        :param candidates: list of all legal moves in the position
        :param color: color of the player to choose from above candidates

        :return: filtered list of moves that was received as first argument
        """

        raise NotImplementedError
