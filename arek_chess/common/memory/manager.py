# -*- coding: utf-8 -*-
from struct import pack, unpack
from typing import List, Dict, Tuple, Optional

from larch.pickle.pickle import dumps, loads
from numpy import float32, ndarray

from arek_chess.board.board import Board
from arek_chess.common.memory.adapters.shared_memory_adapter import SharedMemoryAdapter
from arek_chess.common.memory.base_memory import BaseMemory
from arek_chess.criteria.evaluation.base_eval import ActionType


class MemoryManager:
    """
    Manages the shared memory between multiple processes.

    WARNING! larch library leaks memory, but is very fast, so still is the best choice here.
    In order to release the memory the process using MemoryManager must be killed.
    The leak is small enough to let the engine work until a game is over, then needs a hard restart.
    """

    def __init__(self):
        self.memory: BaseMemory = SharedMemoryAdapter()
        # self.memory: BaseMemory = UltraDictAdapter()

    def get_action(self, size: int) -> ActionType:
        action_bytes = self.memory.get("action")

        action = ndarray(
            shape=(size,), dtype=float32, buffer=action_bytes
        ).tolist()

        return action

    def set_action(self, action: ActionType, size: int) -> None:
        data = ndarray(shape=(size,), dtype=float32)
        data[:] = (*action,)

        self.memory.set("action", data.tobytes())

    def get_node_board(self, node_name: str, board: Optional[Board] = None) -> Optional[Board]:
        board_bytes: Optional[bytes] = self.memory.get(node_name)
        if board_bytes is None:
            return None
        # print(len(board_bytes))

        board = board or Board(fen=None)
        board.deserialize_position(board_bytes)
        return board

        # return loads(board_bytes) if board_bytes is not None else None

    def set_node_board(self, node_name: str, board: Board) -> None:
        # self.memory.set(
        #     node_name, dumps(board, protocol=5, with_refs=False)
        # )

        board_bytes: bytes = board.serialize_position()
        self.memory.set(node_name, board_bytes)

    def get_int(self, key: str) -> int:
        v: bytes = self.memory.get(key)
        return unpack("i", v)[0] if v else 0

    def set_int(self, key: str, value: int, *, new: bool = True) -> None:
        self.memory.set(key, pack("i", value), new=new)

    def get_node_params(self, node_name: str, size: int) -> List[float]:
        params_bytes = self.memory.get(f"{node_name}.params")
        if not params_bytes:
            raise ValueError(f"Not found: {node_name}")

        return ndarray(
            shape=(size,), dtype=float32, buffer=params_bytes
        ).tolist()

    def set_node_params(self, node_name: str, params: List[float]) -> None:
        data = ndarray(shape=(len(params),), dtype=float32)
        data[:] = (*params,)

        self.memory.set(f"{node_name}.params", data.tobytes())

    async def get_node_board_async(self, node_name: str) -> Board:
        board_bytes = await self.memory.get_async(f"{node_name}")
        if not board_bytes:
            raise ValueError(f"Not found: {node_name}")

        return loads(board_bytes)

    def get_many_boards(self, names: List[str]) -> List[Board]:
        boards: List[bytes] = self.memory.get_many([name for name in names])
        return [loads(board) if board is not None else None for board in boards]

    def set_many_boards(self, name_to_board: List[Tuple[str, Board]]):
        name_to_bytes = [
            (name, dumps(board, protocol=5, with_refs=False))
            for name, board in name_to_board
        ]
        self.memory.set_many(name_to_bytes)

    async def set_many_boards_async(self, name_to_board: Dict[str, Board]):
        name_to_bytes = {
            name: dumps(board, protocol=5, with_refs=False)
            for name, board in name_to_board.items()
        }
        await self.memory.set_many_async(name_to_bytes)

    def remove_node_params_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")

    def remove_node_board_memory(self, node_name: str) -> None:
        self.memory.remove(node_name)

    def remove_node_memory(self, node_name: str) -> None:
        self.memory.remove(f"{node_name}.params")
        self.memory.remove(f"{node_name}")

    def clean(self, except_prefix: str = "", silent: bool = False):
        self.memory.clean(except_prefix, silent)
