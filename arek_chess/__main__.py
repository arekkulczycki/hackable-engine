# -*- coding: utf-8 -*-

import argparse

from arek_chess.main.controller import Controller
from arek_chess.main.search_tree_manager import SearchTreeManager

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--clean",
        help="Clean memory for all nodes at given depth and width",
        action="store_true",
        # type=str,
    )
    arg_parser.add_argument(
        "--fen",
        help="Fen of the starting position, defaults to chess starting position.",
        action="store_true",
        # type=str,
    )
    arg_parser.add_argument(
        "--run-once",
        help="Find the best move and quit.",
        action="store_true",
        # type=bool,
    )

    args = arg_parser.parse_args()

    if args.clean:
        width, depth = args.clean.split(",")
        SearchTreeManager.run_clean(int(width), int(depth))
        exit(0)

    controller = Controller()
    controller.boot_up()

    if args.run_once:
        controller.tree_manager.search()
        controller.tear_down()
        exit(0)

    try:
        quitting = False
        while not quitting:
            key = input("choose action... (type help for available actions)\n")
            if key == "q":
                quitting = True
            elif key == "help":
                print("actions:\n")
            elif key == "restart":
                fen = input("type starting fen:\n")
                controller.boot_up(fen)
            elif key == "move":
                controller.make_move()
    finally:
        controller.tear_down()

