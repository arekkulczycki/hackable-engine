from argparse import ArgumentParser
from random import choices, choice
from time import perf_counter

import numpy as np
import torch as th

from hackable_engine.board.hex.bitboard_utils import generate_cells
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.board.hex.move import Move
from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.algorithms.simple_dqn import board_size
from hackable_engine.training.device import Device
from hackable_engine.training.models.graph_gin import GraphGIN
from hackable_engine.training.models.graph_sg import GraphSG
from hackable_engine.training.models.graph_rgcn import GraphRGCN
from hackable_engine.training.models.graph_gmm import GraphGMM
from hackable_engine.training.models.graph_gat import GraphGAT

BOARD_SIZE = 11
MAX_MOVES = BOARD_SIZE**2


parser = ArgumentParser()
parser.add_argument(
    "-v", "--version", type=int, help="version of the model to evaluate", required=True
)
parser.add_argument(
    "-c",
    "--color",
    type=int,
    required=True,
    help="which color player should be trained",
)
args = parser.parse_args()

board = TrainingHexBoard("", size=BOARD_SIZE, use_graph=True)
model = (
    GraphGMM(
        node_count=board.size_square,
        node_features=9,
        output_size=1,
        batch_size=169,
        num_envs=128,
        # num_epochs=epochs,
        gnn_shape=(54, 108, 216, 324, 432, 486),
        # gnn_heads=6,
        mlp_shape=(256,),
        edge_index=board.edge_index,
        # edge_types=board.edge_types,
        pseudo_coordinates=board.pseudo_coordinates,
        # use_res=True,
    )
    .to(Device.XPU)
    .to(th.float32)
)
model.eval()
model_weights = th.load(
    f"dqn/{board_size}/dqn-model-{'white' if args.color else 'black'}.v{args.version}", weights_only=True
)
model.load_state_dict(
    {k.replace("_orig_mod.", ""): v for k, v in model_weights.items()}
)


def select_model_move(board: HexBoard) -> Move:
    logits = model(th.from_numpy(board.get_hetero_graph_node_features_one_hot().reshape((1, 121, 9)).astype(FLOAT_TYPE)).to(Device.XPU))
    logits = logits.reshape((121,))

    illegal_squares = list(generate_cells(board.occupied))
    min_val = logits.argmin().item()
    logits[illegal_squares] = logits[min_val].item()

    mask = 1 << logits.argmax().item()
    return Move(mask, BOARD_SIZE)


def select_opponent_move(board: HexBoard) -> Move:
    opp_color = not args.color
    best_move: Move | None = None
    best_score = None
    n_moves = len(board.move_stack)
    for move in board.legal_moves:
        score = get_distance_score(board, n_moves, perf=True)
        if best_move is None or (
            (
                (opp_color and score > best_score)
                or (not opp_color and score < best_score)
            )
        ):
            best_move = move
            best_score = score
    return best_move


def select_random_move(board):
    moves = list(board.legal_moves)
    return choice(moves)


def get_distance_score(board: HexBoard, n_moves: int, *, perf: bool = False) -> FLOAT_TYPE:
    get_short_missing_distances = board.get_short_missing_distances_perf if perf else board.get_short_missing_distances
    (
        white_missing,
        white_variants,
    ) = get_short_missing_distances(
        True, should_subtract=(not args.color and n_moves % 2 == 1)
    )  # subtracts distance from white because has 1 stone less on board, on odd moves
    (
        black_missing,
        black_variants,
    ) = get_short_missing_distances(False)

    white_score = sum(
        (weight_distance(BOARD_SIZE - k, n_moves) * v) for k, v in white_variants.items()
    )
    black_score = sum(
        (weight_distance(BOARD_SIZE - k, n_moves) * v) for k, v in black_variants.items()
    )

    if not black_score:
        return FLOAT_TYPE(1)
    return np.tanh((white_score / black_score) - 1).astype(FLOAT_TYPE)


def weight_distance(distance, n_moves) -> int:
    if n_moves > MAX_MOVES / 2:
        return distance**3 / MAX_MOVES
    elif n_moves > MAX_MOVES / 4:
        return distance**2 / BOARD_SIZE
    else:
        return distance


def evaluate():
    wins = 0
    losses = 0
    for i in range(10):
        board.reset()
        # TODO: if running more than 10 iterations maybe use random opening move
        board.push(Move.from_coord("a3", 11))
        total = wins + losses
        win_percentage = wins / total if total else 0.5
        random_moves_played = 0
        while not board.is_game_over():
            if board.turn == args.color:
                board.push(select_model_move(board))
                # board.push(select_random_move(board))
            else:
                random_move_weight = (1 - win_percentage) * 0.5 + 0.01
                if choices([True, False], weights=(random_move_weight, 1 - random_move_weight))[0]:
                    board.push(select_random_move(board))
                    random_moves_played += 1
                else:
                    board.push(select_opponent_move(board))

        if board.winner() == args.color:
            wins += 1
        else:
            losses += 1
        # print(board.distance_missing_cached.cache_info())
        print(total, random_moves_played)

    print(f"wins: {wins}, losses: {losses}")


def test_game_speed():
    t0 = perf_counter()
    moves = 0
    while not board.is_game_over():
        if board.turn == args.color:
            board.push(select_random_move(board))
        else:
            moves += 1
            board.push(select_opponent_move(board))

    t = perf_counter() - t0
    print(f"game finished in {t} with {moves} calculated moves, speed {moves/t} moves per second")


if __name__ == "__main__":
    evaluate()
