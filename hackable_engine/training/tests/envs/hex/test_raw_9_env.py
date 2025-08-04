from random import random
from time import perf_counter
from unittest import TestCase

from hackable_engine.training.envs.hex.raw_9_env import Raw9Env


class TestRaw9Env(TestCase):

    @classmethod
    def perf_test(cls):
        t0 = perf_counter()
        env = Raw9Env(models=[None])
        env.reset()
        loops = 0
        while loops < 36:
            obs, reward, finished, _, data = env.step([random()])
            if finished:
                loops += 1
        print(f"finished perf test in {perf_counter() - t0}")


if __name__ == "__main__":
    TestRaw9Env.perf_test()
