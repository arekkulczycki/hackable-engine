# -*- coding: utf-8 -*-
"""
Receive commands and run the engine.
"""

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())  # instead of writing PYTHONPATH=.

from hackable_engine.ui.ui import UI  # pylint: disable=wrong-import-position


def run_with_args():
    """Run the engine with given CLI arguments"""

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-m",
        "--move",
        help="Find the best move and quit.",
        action="store_true",
    )
    arg_parser.add_argument(
        "-n",
        "--notation",
        help="Notation of the starting position, defaults to normal starting position.",
    )
    arg_parser.add_argument(
        "-mv",
        "--model-version",
        help="Name of the trained model file in root directory to be used for evalueation.",
    )
    arg_parser.add_argument("-T", "--timeout", type=float, help="Timeout")
    arg_parser.add_argument("-G", "--game", type=str, choices=["chess", "hex"], help="Game to be played", required=True)
    arg_parser.add_argument(
        "-S",
        "--board-size",
        required="-G=hex" in sys.argv,
        type=int,
        help="Size of the board to be played on.",
    )

    args = arg_parser.parse_args()

    ui = UI(
        position=args.notation,
        game=args.game,
        board_size=args.board_size
    )

    if args.move:
        print(ui.get_move())
    else:
        ui.run()


if __name__ == "__main__":
    run_with_args()
