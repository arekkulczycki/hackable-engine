# -*- coding: utf-8 -*-
"""
Adds an option to run code profiling over a process using this mixin.
"""

import sys
from signal import signal, SIGTERM

from pyinstrument import Profiler


class ProfilerMixin:
    """
    Adds an option to run code profiling over a process using this mixin.
    """

    def _profile_code(self) -> None:
        """"""

        profiler = Profiler()
        profiler.start()

        # tracemalloc.start()

        def before_exit(*_) -> None:
            """"""

            # print(f"call count: {self.call_count}")
            profiler.stop()
            profiler.print(show_all=True)

            # snapshot = tracemalloc.take_snapshot()
            # top_stats = snapshot.statistics('lineno')
            # print("[ Top 10 ]")
            # for stat in top_stats[:10]:
            #     print(stat)
            # tracemalloc.stop()

            sys.exit(0)

        signal(SIGTERM, before_exit)
