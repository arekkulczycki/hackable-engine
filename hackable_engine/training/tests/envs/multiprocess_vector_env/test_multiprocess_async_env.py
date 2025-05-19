import asyncio
from random import randint, choice
from time import perf_counter

import numpy as np
from gymnasium.vector import SyncVectorEnv

from hackable_engine.board.hex.bitboard_utils import generate_cells
from hackable_engine.common.constants import FLOAT_TYPE
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
from hackable_engine.training.envs.multiprocess_vector_env.multiprocess_async_env import MultiprocessAsyncEnv
from hackable_engine.training.envs.wrappers.episode_stats import EpisodeStats

BOARD_MASK = (2**169)-1
num_workers = 8
num_envs = 128


def get_env():
    return MultiprocessAsyncEnv(
        lambda seed, num_envs, color_, models=[]: EpisodeStats(
            SyncVectorEnv(
                [
                    lambda: Logit13GraphEnv(color=False, models=[None])
                    for _ in range(num_envs)
                ],
                copy=False,
            ),
            is_multiprocessed=True,
        ),
        num_workers,
        int(num_envs // num_workers),
        action_shape=(1,),
        color=False,
    )


async def benchmark():
    env = get_env()

    total = 0
    t00 = perf_counter()
    _, ocbs, ocws = env.reset()
    total += perf_counter() - t00

    steps = 0
    while steps < 10_000:
        arr = np.empty((num_envs,), dtype=np.int32)
        for i in range(num_envs):
            ocb = ocbs[i]
            ocw = ocws[i]
            un_oc = BOARD_MASK ^ (ocb | ocw)
            legal_moves = list(generate_cells(un_oc))
            arr[i] = choice(legal_moves)
        t0 = perf_counter()
        _, _, _, _, ocbs, ocws = await env.step(arr)
        total += perf_counter() - t0
        steps += num_envs

    print(f"finished perf test in {perf_counter() - t00}", f"{steps / total} steps per second")


if __name__ == "__main__":
    asyncio.run(benchmark())
