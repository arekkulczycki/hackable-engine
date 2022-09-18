import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
# print(os.getcwd())
# print(os.path.dirname(__file__))
# exit(1)
from arek_chess.main.controller import Controller
from arek_chess.main.game_tree.search_manager import SearchManager

if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c",
        "--clean",
        help="Clean memory for all nodes at given depth and width",
        # type=str,
    )
    arg_parser.add_argument(
        "--fen",
        help="Fen of the starting position, defaults to chess starting position.",
        # type=str,
    )
    arg_parser.add_argument(
        "--run-once",
        help="Find the best move and quit.",
        action="store_true",
    )

    args = arg_parser.parse_args()

    if args.clean:
        width, depth = args.clean.split(",")
        SearchManager.run_clean(int(width), int(depth))
        exit(0)

    controller = Controller()
    controller.boot_up()

    if args.run_once:
        controller.search_manager.search()
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
            try:
                n = int(key)
                for i in range(n):
                    controller.make_move()
            except:
                pass
    finally:
        controller.tear_down()
