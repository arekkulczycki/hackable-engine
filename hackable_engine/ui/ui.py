# -*- coding: utf-8 -*-
from typing import Optional, TypeVar, Generic, Callable, Dict

from hackable_engine.board import GameBoardBase
from hackable_engine.common.constants import Game
from hackable_engine.controller import Controller
from hackable_engine.criteria.evaluation.base_eval import WeightsType
from hackable_engine.workers.configs.search_tree_cache import SearchTreeCache

GameBoardT = TypeVar("GameBoardT", bound=GameBoardBase)

CONTOLLER_CONFIGURATION_METHODS: Dict[Game, Callable] = {
    Game.CHESS: Controller.configure_for_chess,
    Game.HEX: Controller.configure_for_hex
}


class UI(Generic[GameBoardT]):
    """
    Provides a layer of communication to the outside world. Requires adapters to translate to UCI or other interfaces.
    """

    def __init__(self, game: Game, board_size: int, position: str) -> None:
        self.controller = CONTOLLER_CONFIGURATION_METHODS[game](position=position, board_size=board_size)
        self.controller.boot_up()

    def run(self) -> None:
        """"""

        quitting: bool = False  # pylint: disable=invalid-name
        while not quitting:
            key = input("choose action... (type help for available actions)\n")
            if key == "q":
                # teardown done on `finally`
                quitting = True
            elif key == "help":
                print("actions:\n")
            elif key == "restart":
                fen = input("type starting fen:\n")
                self.controller.reset(fen)
            elif key == "move":
                self.get_move()
            elif key == "game":
                self.game()

            try:
                n = int(key)
                for i in range(n):
                    self.make_move()
            except TypeError:
                print("unknown command")

    def game(self) -> None:
        """"""

        while not self.controller.board.is_game_over():
            self.make_move()
            print(self.controller.board.position())

        winner = self.controller.board.winner()
        result = "draw" if winner is None else "white won" if winner else "black won"
        print(f"game over, result: {result}")

    def make_move(
        self,
        memory_action: Optional[WeightsType] = None,
        cached_search_tree: Optional[SearchTreeCache] = None,
    ) -> None:
        """"""

        self.controller.setup_search_worker(search_tree=cached_search_tree)

        self.controller.board.push_coord(self.get_move(memory_action))

    def get_move(
        self,
        memory_action: Optional[WeightsType] = None,
    ) -> str:
        """"""

        if self.controller.board.is_game_over():
            print(self.controller.board.get_notation())
            raise ValueError("asked for a move in a game over position")

        if memory_action is not None:
            with self.controller.worker_locks.weights_lock:
                self.controller.memory_manager.set_action(memory_action, len(memory_action))

        return self.controller.search()

    def __del__(self):
        """"""

        self.controller.tear_down()
