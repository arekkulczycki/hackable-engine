"""
Receive commands and run the engine.
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # instead of writing PYTHONPATH=.
# print(os.getcwd())
# print(os.path.dirname(__file__))
# exit(1)
from arek_chess.controller import Controller


def validate_tree_params(tree_params: str) -> None:
    params = tree_params.split(",")
    assert len(params) == 3
    int(params[0])
    int(params[1])


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m",
        "--move",
        help="Find the best move and quit.",
        action="store_true",
    )
    arg_parser.add_argument(
        "-g",
        "--game",
        help="Play entire game and quit.",
        action="store_true",
    )
    arg_parser.add_argument(
        "-c",
        "--clean",
        help="Clean memory for all nodes at given depth and width",
        action="store_true",
    )
    arg_parser.add_argument(
        "-f",
        "--fen",
        help="Fen of the starting position, defaults to chess starting position.",
    )
    arg_parser.add_argument(
        "-p",
        "--printing",
        type=int,
        default=0,
        help="0 - nothing, 1 - top candidates, 2 - entire searched tree (look at --tree-params)",
    )
    arg_parser.add_argument(
        "-tp",
        "--tree-params",
        default="3,5,",
        help="3 values split by comma: min_depth, max_depth, candidate. Example: --tree-params=3,7,f3e5",
    )
    arg_parser.add_argument(
        "-l",
        "--search-limit",
        type=int,
        help="The engine will look at roughly 2^<LIMIT> nodes. "
        "14-15 is suggested for quick results. "
        "16-18 for a thorough examination.",
    )
    arg_parser.add_argument(
        "-mv",
        "--model-version",
        help="Name of the trained model file in root directory to be used for evalueation.",
    )
    arg_parser.add_argument("-t", "--timeout", type=float, help="Timeout")
    arg_parser.add_argument(
        "-th",
        "--thread",
        help="Runs in a new thread.",
        action="store_true",
    )

    args = arg_parser.parse_args()

    if args.clean:
        Controller().release_memory()
        sys.exit(0)

    validate_tree_params(args.tree_params)
    controller = Controller(
        position=args.fen,
        printing=args.printing,
        tree_params=args.tree_params,
        search_limit=args.search_limit,
        model_version=args.model_version,
        timeout=args.timeout,
    )
    controller.boot_up()

    if args.move:
        controller.make_move()
        controller.stop_child_processes()
        sys.exit(0)
    elif args.game:
        controller.play()
        sys.exit(0)

    try:
        quitting = False
        while not quitting:
            key = input("choose action... (type help for available actions)\n")
            if key == "q":
                # teardown done on `finally`
                quitting = True
            elif key == "help":
                print("actions:\n")  # TODO: ???
            elif key == "clean":
                controller.release_memory()
            elif key == "restart":
                fen = input("type starting fen:\n")
                controller.reset_board(fen)
            elif key == "move":
                controller.make_move()
            elif key == "play":
                controller.play()
            # elif key == "pgn":
            #     print(controller.get_pgn())
            try:
                n = int(key)
                for i in range(n):
                    controller.make_move()
            except:
                pass
    finally:
        controller.tear_down()
