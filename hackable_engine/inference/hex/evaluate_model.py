from argparse import ArgumentParser
from random import choices, choice
from time import perf_counter

import numpy as np

from hackable_engine.board.hex.bitboard_utils import generate_cells
from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.board.hex.move import Move
from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.algorithms.simple_dqn import board_size
from hackable_engine.training.utils.device import Device

import onnxruntime as ort

BOARD_SIZE = 13
MAX_MOVES = BOARD_SIZE**2
# fmt: on
if board_size == 11:
    OPENINGS = [
        "a3","a4","a5","a6","a7","a8","a9","a10","a11",
        "k1","k2","k3","k4","k5","k6","k7","k8","k9",
        "c2","c10","d2","d10","e2","e10","f2","f10","g2","g10","h2","h10","i2","i10",
        "e3","e9","f3","f9","g3","g9"
    ]
elif board_size == 13:
    OPENINGS = [
        "a1","a2","a3","a4","a5","a6","a7","a8","a9","a10","a11","a12","a13",
        "m1","m2","m3","m4","m5","m6","m7","m8","m9","m10","m11","m12","m13",
        "c2","c12","d2","d12","e2","e12","f2","f12","g2","g12","h2","h12","i2","i12","k2","k12",
        "b2","l12","f3","g3","h3","i3","f11","g11","h11","i11",
    ]  # 52 openings
else:
    raise NotImplementedError
# fmt: off

parser = ArgumentParser()
parser.add_argument("-v", "--version", type=int, help="version of the model to evaluate", required=True)
parser.add_argument("-c", "--color", type=int, required=True, help="which color is evaluated")
parser.add_argument("-i", "--iterations", type=int, default=10, help="number of games to play")
parser.add_argument(
    "-p", "--performance", type=int, default=1, help="faster but weaker"
)
args = parser.parse_args()
print("version:", args.version, ", color:", args.color, ", iterations:", args.iterations, ", perf:", bool(args.performance))

device = Device.CPU

board = TrainingHexBoard("", size=BOARD_SIZE)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_cpu = ort.InferenceSession(
    f"{board_size}_{'white' if args.color else 'black'}_{args.version}.onnx", providers=["CPUExecutionProvider"], sess_options=sess_options
)


def select_model_move(board: HexBoard) -> Move:
    obs = board.get_hetero_graph_node_features_one_hot().reshape((1, board_size**2, 9)).astype(FLOAT_TYPE)
    # logits = model(th.from_numpy(obs).to(device))
    logits = ort_session_cpu.run(None, {"inputs": obs})[0]
    logits = logits.reshape((board_size**2,))
    # print(logits)

    illegal_squares = list(generate_cells(board.occupied))
    min_val = logits.argmin().item()
    logits[illegal_squares] = logits[min_val].item()

    return Move.from_c(logits.argmax().item(), size=BOARD_SIZE)


def select_opponent_move(board: HexBoard) -> Move:
    opp_color = not args.color
    best_move: Move | None = None
    best_score = None
    n_moves = len(board.move_stack)
    for move in board.legal_moves:
        score = get_distance_score(board, n_moves, perf=bool(args.performance))
        if best_move is None or (((opp_color and score > best_score) or (not opp_color and score < best_score))):
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

    white_score = sum((weight_distance(BOARD_SIZE - k, n_moves) * v) for k, v in white_variants.items())
    black_score = sum((weight_distance(BOARD_SIZE - k, n_moves) * v) for k, v in black_variants.items())

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
    for i in range(args.iterations):
        board.reset()
        # TODO: if running more than 10 iterations maybe use random opening move
        board.push(Move.from_coord(choice(OPENINGS), board_size))
        total = wins + losses
        minimum_logical_moves = (0.8 if bool(args.performance) else 0.9)
        win_percentage = wins / total if total else minimum_logical_moves
        random_moves_played = 0
        while not board.is_game_over():
            if board.turn == args.color:
                board.push(select_model_move(board))
            else:
                random_move_weight = (1 - win_percentage)**2 * (1 - minimum_logical_moves)
                if choices([True, False], weights=(random_move_weight, 1 - random_move_weight))[0]:
                    board.push(select_random_move(board))
                    random_moves_played += 1
                else:
                    board.push(select_opponent_move(board))

        if board.winner() == args.color:
            wins += 1
            result = "win"
        else:
            losses += 1
            result = "loss"
        # print(board.distance_missing_cached.cache_info())
        print(total, result, f"played {random_moves_played} random moves", board.get_notation())

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
