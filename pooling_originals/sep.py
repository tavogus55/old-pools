from typing import Callable
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, Size
from torch_sparse import SparseTensor, matmul

class SEPooling(MessagePassing):
    def __init__(self, nn: Callable, **kwargs):
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'target_to_source')
        super().__init__(**kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(self, x: Tensor, edge_index: Adj, size: Size = None) -> Tensor:
        out = self.propagate(edge_index, x=x, size=size)
        return out
        # return self.nn(out)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def message_and_aggregate(self, adj_t: SparseTensor,
                              x: Tensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

