"""
Manages the shared memory between multiple processes.
"""

from typing import List, Dict, Tuple, Optional

import numpy
from larch.pickle.pickle import dumps, loads

from arek_chess.board.board import Board
from arek_chess.common.memory.base_memory import BaseMemory
from arek_chess.common.memory.shared_memory import SharedMemory
from arek_chess.criteria.evaluation.base_eval import BaseEval


class MemoryManager:
    """
    Manages the shared memory between multiple processes.

    WARNING! larch library leaks memory, but is very fast, so still is the best choice here.
    In order to release the memory the process using MemoryManager must be killed.
    The leak is small enough to let the engine work until a game is over, then needs a hard restart.
    """

    def __init__(self):
        self.memory: BaseMemory = SharedMemory()
        # self.memory: BaseMemory = RedisMemory()

    def get_action(self, size: int) -> BaseEval.ActionType:
        action_bytes = self.memory.get("action")

        action = numpy.ndarray(
            shape=(size,), dtype=numpy.double, buffer=action_bytes
        ).tolist()

        return action

    def set_action(self, action: BaseEval.ActionType, size: int) -> None:
        data = numpy.ndarray(shape=(size,), dtype=numpy.double)
        data[:] = (*action,)

        self.memory.set("action", data.tobytes())

    def get_dispatched(self) -> int:
        dispatched: Optional[bytes] = self.memory.get("dispatched", b"0")

        return int(dispatched.decode()) if dispatched is not None else 0

    def set_dispatched(self, dispatched: int) -> None:
        self.memory.set("dispatched", str(dispatched).encode())

    def get_node_params(self, node_name: str, size: int) -> List[float]:
        params_bytes = self.memory.get(f"{node_name}.params")
        if not params_bytes:
            raise ValueError(f"Not found: {node_name}")

        return numpy.ndarray(
            shape=(size,), dtype=numpy.float16, buffer=params_bytes
        ).tolist()

    def get_node_board(self, node_name: str) -> Optional[Board]:
        board_bytes: Optional[bytes] = self.memory.get(f"{node_name}")
        # print(len(board_bytes))

        return loads(board_bytes) if board_bytes is not None else None

    async def get_node_board_async(self, node_name: str) -> Board:
        board_bytes = await self.memory.get_async(f"{node_name}")
        if not board_bytes:
            raise ValueError(f"Not found: {node_name}")

        return loads(board_bytes)

    def set_node_params(self, node_name: str, params: List[float]) -> None:
        data = numpy.ndarray(shape=(len(params),), dtype=numpy.float16)
        data[:] = (*params,)

        self.memory.set(f"{node_name}.params", data.tobytes())

    def set_node_board(self, node_name: str, board: Board) -> None:
        self.memory.set(
            f"{node_name}", dumps(board, protocol=5, with_refs=False)
        )

    async def set_node_board_async(self, node_name: str, board: Board) -> None:
        await self.memory.set_async(
            f"{node_name}", dumps(board, protocol=5, with_refs=False)
        )

    def get_many_boards(self, names: List[str]) -> List[Board]:
        boards: List[bytes] = self.memory.get_many([f"{name}" for name in names])
        return [loads(board) if board is not None else None for board in boards]

    def set_many_boards(self, name_to_board: List[Tuple[str, Board]]):
        name_to_bytes = [
            (f"{name}", dumps(board, protocol=5, with_refs=False))
            for name, board in name_to_board
        ]
        self.memory.set_many(name_to_bytes)

    async def set_many_boards_async(self, name_to_board: Dict[str, Board]):
        name_to_bytes = {
            f"{name}": dumps(board, protocol=5, with_refs=False)
            for name, board in name_to_board.items()
        }
        await self.memory.set_many_async(name_to_bytes)

    def remove_node_params_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")

    def remove_node_board_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}")

    def remove_node_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")
        self.memory.remove(f"{node_name}")

    def clean(self):
        self.memory.clean()
