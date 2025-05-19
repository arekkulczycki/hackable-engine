from random import randint, choice
from time import perf_counter

from hackable_engine.board.hex.bitboard_utils import generate_cells
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv

BOARD_MASK = (2**169)-1


def benchmark():
    env = Logit13GraphEnv(models=[None])

    t0 = perf_counter()
    obs, data = env.reset()
    loops = 0
    steps = 0
    while loops < 3:
        un_oc = BOARD_MASK ^ (int.from_bytes(data["ocb"]) | int.from_bytes(data["ocw"]))
        legal_moves = list(generate_cells(un_oc))
        obs, reward, finished, _, data = env.step(choice(legal_moves))
        steps += 1
        if finished:
            obs, data = env.reset()
            print(loops)
            loops += 1

    t = perf_counter() - t0
    print(f"finished perf test in {t}", f"{steps / t} steps per second")


if __name__ == "__main__":
    benchmark()
