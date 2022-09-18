"""
Manages the shared memory between multiple processes.
"""

from typing import List, Dict

import numpy
from larch import pickle

from arek_chess.board.board import Board
from arek_chess.utils.memory.base_memory import BaseMemory
from arek_chess.utils.memory.redis_memory import RedisMemory


class MemoryManager:
    """
    Manages the shared memory between multiple processes.
    """

    def __init__(self):
        # self.memory: BaseMemory = SharedMemory()
        self.memory: BaseMemory = RedisMemory()

    def get_node_params(self, node_name: str, size: int) -> List[float]:
        params_bytes = self.memory.get(f"{node_name}.params")
        if not params_bytes:
            raise ValueError(f"Not found: {node_name}")

        return numpy.ndarray(
            shape=(size,), dtype=numpy.float16, buffer=params_bytes
        ).tolist()

    def get_node_board(self, node_name: str) -> Board:
        board_bytes = self.memory.get(f"{node_name}.board")
        if not board_bytes:
            raise ValueError(f"Not found: {node_name}")

        return pickle.loads(board_bytes)

    def set_node_params(self, node_name: str, params: List[float]) -> None:
        data = numpy.ndarray(shape=(len(params),), dtype=numpy.float16)
        data[:] = (*params,)

        self.memory.set(f"{node_name}.params", data.tobytes())

    def set_node_board(self, node_name: str, board: Board) -> None:
        self.memory.set(
            f"{node_name}.board", pickle.dumps(board, protocol=5, with_refs=False)
        )

    def get_many_boards(self, names: List[str]) -> List[Board]:
        boards: List[bytes] = self.memory.get_many([f"{name}.board" for name in names])
        return [pickle.loads(board) for board in boards if board]

    def set_many_boards(self, name_to_board: Dict[str, Board]):
        name_to_bytes = {
            f"{name}.board": pickle.dumps(board, protocol=5, with_refs=False)
            for name, board in name_to_board.items()
        }
        self.memory.set_many(name_to_bytes)

    def remove_node_params_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")

    def remove_node_board_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.board")

    def remove_node_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")
        self.memory.remove(f"{node_name}.board")
