# -*- coding: utf-8 -*-
from typing import Optional

from numpy import array, eye, float32, int8, ndarray, reshape
from js import ort, globalThis
from pyodide.ffi import to_js

from hackable_engine.board.hex.hex_board import HexBoard
from hackable_engine.criteria.evaluation.base_eval import WeightsType, BaseEval


class WasmEval(BaseEval[HexBoard]):
    """"""

    PARAMS_NUMBER: int = 8

    def __init__(self, size: int):
        """"""

        self.size = size

    def get_score(
        self, board: HexBoard, is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        """"""

        obs = self.observation_from_board(board)
        js_input = ort.Tensor.new("float32", to_js(obs.ravel()), to_js(obs.shape))
        if board.turn:
            value = globalThis.ortSessionWhite.run(input=js_input)
        else:
            value = globalThis.ortSessionBlack.run(input=js_input)
        print(value)
        return array(value["output"].data.to_py()).item()

    @staticmethod
    def observation_from_board(board: HexBoard) -> ndarray:
        """"""

        return board.get_hetero_graph_node_features_one_hot()

    def get_scores(
            self, boards: list[HexBoard], is_check: bool, weights: Optional[WeightsType] = None
    ) -> float32:
        pass
        # TODO: run inference on multiple boards at once
