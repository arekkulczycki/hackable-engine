import numpy
import torch
import onnxruntime as ort

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard
from hackable_engine.training.device import Device
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.models.graph_gmm import GraphGMM

torch_path = "model.pt"
onnx_path = "model.onnx"
obs_sample = Logit11GraphEnv.observation_space.sample().astype(numpy.float32).reshape(1, 121, 9)
obs_sample_torch = torch.from_numpy(obs_sample).to(torch.float32)

board = TrainingHexBoard(size=11, use_graph=True)
model = GraphGMM(
    node_count=board.size_square,
    node_features=9,
    output_size=1,
    batch_size=64,
    num_envs=128,
    gnn_shape=(54, 108, 216, 324, 432, 486),
    # gnn_heads=6,
    mlp_shape=(256,),
    edge_index=board.edge_index,
    # edge_types=board.edge_types,
    # edge_types=board.edge_types_rgcn,
    pseudo_coordinates=board.pseudo_coordinates,
    # use_res=True,
    device=Device.CPU,
)
state_dict = torch.load(torch_path, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    torch_output = model(obs_sample_torch).numpy()


torch.onnx.export(
    model,
    obs_sample_torch,
    onnx_path,
    opset_version=17,
    input_names=["inputs"],
    dynamic_axes={
        "inputs": {0: "input"},
        "actions": [0]
    },
    # export_params=True,
    # opset_version=11,
    # do_constant_folding=True,
    # input_names=['input'],
    # output_names=['output'],
    # dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
print(f"Model exported to {onnx_path}")

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
ort_session_cpu = ort.InferenceSession(
    onnx_path, providers=["CPUExecutionProvider"], sess_options=sess_options
)
ort_output = ort_session_cpu.run(None, {"inputs": obs_sample})[0]


print(torch_output, ort_output)
are_close = numpy.allclose(torch_output, ort_output, rtol=1e-03, atol=1e-05)
print("Are PyTorch and ONNX outputs close?", are_close)