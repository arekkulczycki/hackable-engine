import torch as th
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch_geometric.utils import spmm


class GraphConvolution(Module):
    """
    Graph convolution layer from https://github.com/Gkunnan97/FastGCN_pytorch/blob/gpu/layers.py.
    """

    def __init__(self, in_features, out_features, bias=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(th.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(th.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x, adj):
        # print(self.weight)
        support = th.matmul(x, self.weight)
        output = spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
