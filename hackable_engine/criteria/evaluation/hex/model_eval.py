# -*- coding: utf-8 -*-
from typing import Optional

from nptyping import Int8, NDArray, Shape
from numpy import eye, float32, int8, reshape
from onnxruntime import InferenceSession

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import ActionType, BaseEval


class ModelEval(BaseEval[HexBoard]):
    """"""

    ACTION_SIZE: int = 8

    def __init__(self, size: int, model_path: str):
        """"""

        self.size = size
        self.model_path = model_path
        self.ort_session = InferenceSession(model_path, providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider'])

    def get_score(
        self, board: HexBoard, is_check: bool, action: Optional[ActionType] = None
    ) -> float32:
        """"""

        action, value = self.ort_session.run(None, {"input": self.observation_from_board(board)})
        return action

    def observation_from_board(self, board: HexBoard) -> NDArray:
        """"""

        local: NDArray[Shape, Int8] = board.get_neighbourhood(
            self.size, should_suppress=True
        )
        # fmt: off
        obs = eye(3, dtype=int8)[local][:, 1:].flatten()  # dummy encoding - 2 columns of 0/1 values, 1 column dropped
        return reshape(obs.astype(float32), (1, self.size**2))
        # fmt: on
