from typing import Optional

from numpy import expand_dims, eye, float32, int8, mean, ndarray, partition, reshape
from onnxruntime import InferenceSession
# TODO: use openvino, native openvino is faster than onnx on openvino backend

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval


class ModelEval(BaseEval[HexBoard]):
    """"""

    PARAMS_NUMBER: int = 8

    def __init__(self, size: int):
        """"""

        self.size = size
        self.ort_session_black = InferenceSession("11black.onnx", providers=["OpenVINOExecutionProvider"])
        self.ort_session_white = InferenceSession("11white.onnx", providers=["OpenVINOExecutionProvider"])

    def get_score(
        self, board: HexBoard, is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        """"""

        if board.turn:
            logits = self.ort_session_black.run(None, {"inputs": self.observation_from_board(board)})
            # take 3 largest values from the output (logits)
            top3 = partition(logits, -3)[-3:]
        else:
            logits = self.ort_session_white.run(None, {"inputs": self.observation_from_board(board)})
            # take 3 lowest values from the output (logits)
            top3 = partition(logits, 2)[:3]
        return mean(top3)

    @staticmethod
    def observation_from_board(board: HexBoard) -> ndarray:
        """"""

        return expand_dims(board.get_hetero_graph_node_features_one_hot(), axis=0)

    def get_scores(
            self, boards: list[HexBoard], is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        pass
        # TODO: run inference on multiple boards at once
