# -*- coding: utf-8 -*-
import sys
from multiprocessing import Process

from arek_chess.common.memory.adapters.shared_memory_adapter import remove_shm_from_resource_tracker
from arek_chess.common.memory.manager import MemoryManager
from arek_chess.common.profiler_mixin import ProfilerMixin
from arek_chess.criteria.evaluation.base_eval import ActionType


class BaseWorker(Process, ProfilerMixin):
    """
    Base for the worker process.
    """

    def run(self) -> None:
        """"""

        try:
            self.memory_manager: MemoryManager = MemoryManager()
            remove_shm_from_resource_tracker()

            self._run()
        except KeyboardInterrupt:
            sys.exit(0)

    def _run(self) -> None:
        """"""

        raise NotImplementedError

    def get_memory_action(self, size: int) -> ActionType:
        """"""

        return tuple(self.memory_manager.get_action(size))
