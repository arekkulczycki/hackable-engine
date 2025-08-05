from sys import argv

import numpy
import numpy as np
import torch
import onnxruntime as ort
from torch.export import Dim

from hackable_engine.board.hex.training.training_hex_board import TrainingHexBoard
from hackable_engine.training.algorithms.simple_dqn import GNN_SHAPES
from hackable_engine.training.envs.hex.logit_13_graph_env import Logit13GraphEnv
from hackable_engine.training.models.graph_gat import GraphGAT
from hackable_engine.training.models.graph_gatformer import GraphGATformer
from hackable_engine.training.models.graph_gatgin import GraphGATGIN
from hackable_engine.training.models.graph_gin import GraphGIN
from hackable_engine.training.models.graph_gine import GraphGINE
from hackable_engine.training.models.graph_gmmformer import GraphGMMformer
from hackable_engine.training.models.graph_rgcn import GraphRGCN
from hackable_engine.training.models.graph_sg import GraphSG
from hackable_engine.training.utils.device import Device
from hackable_engine.training.envs.hex.logit_11_graph_env import Logit11GraphEnv
from hackable_engine.training.models.graph_gmm import GraphGMM


torch_path = argv[1] if len(argv) >= 3 else "model.pt"
onnx_path = argv[2] if len(argv) >= 3 else "model.onnx"
# obs_sample = Logit11GraphEnv.observation_space.sample().astype(numpy.float32).reshape(1, 121, 9)
obs_sample = Logit13GraphEnv.observation_space.sample().astype(numpy.float32).reshape(1, 169, 9)
obs_sample_torch = torch.from_numpy(obs_sample).to(torch.float32)

board = TrainingHexBoard(size=13)
model_class = GraphGMM
model = model_class(
    node_count=board.size_square,
    node_features=9,
    output_size=1,
    batch_size=64,
    dropouts=0.00,
    num_envs=128,
    gnn_shape=GNN_SHAPES[model_class],
    # gnn_heads=(6, 3, 1, 1, 1),
    mlp_shape=(256,),
    edge_index=board.edge_index,
    # edge_types=board.edge_types,
    # edge_types=board.edge_types_rgcn,
    pseudo_coordinates=board.pseudo_coordinates,
    device=Device.CPU,
)
state_dict = torch.load(torch_path, map_location="cpu")
model.load_state_dict(state_dict, strict=True, assign=True)
model = model.to(Device.CPU)
model.eval()

with torch.no_grad():
    torch_output = model(obs_sample_torch).numpy()


torch.onnx.export(
    model,
    obs_sample_torch,
    onnx_path,
    opset_version=20,
    input_names=["inputs"],
    output_names=["output"],
    # export_params=True,
    # do_constant_folding=True,
    dynamic_axes={"inputs": {0: "batch_size"}, "output": {0: "batch_size"}},
    dynamo=False,
    # external_data=False,
    # verify=True,
    # verbose=True,
    # dynamic_shapes={"x": {0: Dim("dim0_x")}},
)
print(f"Model exported to {onnx_path}")

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
ort_session_cpu = ort.InferenceSession(
    onnx_path, providers=["CPUExecutionProvider"], sess_options=sess_options
)
# ort_output = ort_session_cpu.run(None, {"inputs": np.stack([obs_sample.reshape((121, 9)), obs_sample.reshape((121, 9))])})[0]
ort_output = ort_session_cpu.run(None, {"inputs": obs_sample})[0]
print(torch_output.shape, ort_output.shape)
# stacked = np.stack([obs_sample.reshape((121, 9)) for i in range(8)])
# stacked = np.stack([obs_sample for i in range(8)])
# print("stacked shape", stacked.shape)
# ort_session_cpu.run(None, {"inputs": stacked})[0]


print(torch_output, ort_output)
are_close = numpy.allclose(torch_output, ort_output, rtol=1e-03, atol=1e-05)
print("Are PyTorch and ONNX outputs close?", are_close)

# traced_script_module = torch.jit.trace(model, torch.from_numpy(obs_sample))
# traced_script_module.save("traced_model.pt")
# with torch.no_grad():
#     output_torchscript = traced_script_module(torch.from_numpy(obs_sample))
#
# are_close = torch.allclose(torch.from_numpy(torch_output), output_torchscript, rtol=1e-04, atol=1e-06)
# print("Are PyTorch and TorchScript outputs close?", are_close)

# if not are_close:
#     max_index_pytorch = np.argmax(torch_output, axis=1)
#     max_index_onnx = np.argmax(ort_output, axis=1)
#
#     # Check if the indices are the same
#     same_max_index = np.array_equal(max_index_pytorch, max_index_onnx)
#     print("Are the maximum indices the same?", same_max_index)
