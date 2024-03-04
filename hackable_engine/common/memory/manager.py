# -*- coding: utf-8 -*-
from struct import pack, unpack
from typing import Any, Optional

from nptyping import NDArray

from numpy import float32, ndarray

from hackable_engine.board import GameBoardBase
from hackable_engine.common.constants import MEMORY_HANDLER, MemoryHandler
from hackable_engine.common.memory.base_memory import BaseMemory
from hackable_engine.criteria.evaluation.base_eval import WeightsType

if MEMORY_HANDLER == MemoryHandler.SHARED_MEM:
    from hackable_engine.common.memory.adapters.shared_memory_adapter import (
        SharedMemoryAdapter,
    )
elif MEMORY_HANDLER == MemoryHandler.REDIS:
    from hackable_engine.common.memory.adapters.redis_adapter import RedisAdapter
elif MEMORY_HANDLER == MemoryHandler.WASM:
    from hackable_engine.common.memory.adapters.wasm_adapter import WasmAdapter


class MemoryManager:
    """
    Manages the shared memory between multiple processes.

    WARNING! larch library leaks memory, but is very fast, so still is the best choice here.
    In order to release the memory the process using MemoryManager must be killed.
    The leak is small enough to let the engine work until a game is over, then needs a hard restart.
    """

    def __init__(self, memory: Optional[Any] = None):
        # self.memory: BaseMemory = UltraDictAdapter()

        if MEMORY_HANDLER == MemoryHandler.SHARED_MEM:
            self.memory = SharedMemoryAdapter()
        elif MEMORY_HANDLER == MemoryHandler.REDIS:
            self.memory: BaseMemory = RedisAdapter()
        elif MEMORY_HANDLER == MemoryHandler.WASM:
            self.memory: BaseMemory = WasmAdapter(memory)

    def get_weights(self, size: int) -> WeightsType:
        action_bytes = self.memory.get("action")

        return ndarray(shape=(size,), dtype=float32, buffer=action_bytes)

    def set_action(self, action: WeightsType, size: int) -> None:
        data: NDArray = ndarray(shape=(size,), dtype=float32)
        data[:] = (*action,)

        self.memory.set("action", data.tobytes())

    def get_node_board(
        self, node_name: str, board: Optional[GameBoardBase] = None
    ) -> Optional[GameBoardBase]:
        board_bytes: Optional[bytes] = self.memory.get(node_name)
        if board_bytes is None:
            return None
        # print(len(board_bytes))

        board = board or GameBoardBase()
        board.deserialize_position(board_bytes)
        return board

        # return loads(board_bytes) if board_bytes is not None else None

    def set_node_board(self, node_name: str, board: GameBoardBase) -> None:
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

    def get_bool(self, key: str) -> bool:
        v: bytes = self.memory.get(key)
        return unpack("?", v)[0] if v else False

    def set_bool(self, key: str, value: int, *, new: bool = True) -> None:
        self.memory.set(key, b"1" if value else b"0", new=new)

    def get_str(self, key: str) -> str:
        v: bytes = self.memory.get(key)
        return v and v.decode()

    def set_str(self, key: str, value: str, *, new: bool = True) -> None:
        self.memory.set(key, value.encode(), new=new)

    def remove(self, key: str) -> None:
        self.memory.remove(key)

    def clean(self, except_prefix: str = ""):
        self.memory.clean(except_prefix)
