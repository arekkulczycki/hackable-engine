import asyncio
import sys
from multiprocessing import Process

from setproctitle import setproctitle

# from hackable_engine.common.memory.adapters.shared_memory_adapter import remove_shm_from_resource_tracker
from hackable_engine.common.memory.manager import MemoryManager
from hackable_engine.common.profiler_mixin import ProfilerMixin
from hackable_engine.criteria.evaluation.base_eval import WeightsType


class BaseWorker(Process, ProfilerMixin):
    """
    Base for the worker process.
    """

    def __init__(self, name: str, *, memory=None):
        super(Process, self).__init__(None, None, name)

        self.memory_manager: MemoryManager = MemoryManager(memory)

    def run(self) -> None:
        setproctitle(self.name)

        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            sys.exit(0)

    async def _run(self) -> None:
        """"""

        raise NotImplementedError

    def get_memory_action(self, size: int) -> WeightsType:
        """"""

        return tuple(self.memory_manager.get_weights(size))
