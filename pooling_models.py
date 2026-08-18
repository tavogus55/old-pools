import time

import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    SAGPooling,
    TopKPooling,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import subgraph
from torch_scatter import scatter_add


def batched_random_topk(batch, ratio):
    """Select a uniformly random, rounded-ratio subset from each graph."""
    num_nodes = scatter_add(
        batch.new_ones(batch.size(0)),
        batch,
        dim=0,
    )
    keep = (ratio * num_nodes.to(torch.float)).round().to(torch.long).clamp_min_(1)

    # Ranking i.i.d. random scores is uniform sampling without replacement.
    scores = torch.rand(batch.size(0), device=batch.device)
    _, perm = torch.sort(scores, descending=True)
    sorted_batch, batch_order = torch.sort(batch[perm])
    perm = perm[batch_order]

    graph_offsets = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]]
    )
    node_positions = torch.arange(
        batch.size(0), device=batch.device
    ) - graph_offsets[sorted_batch]
    selected = node_positions < keep[sorted_batch]
    return perm[selected]


class NDRPPooling(nn.Module):
    """Uniform node-dropping pooling with a sparse induced subgraph."""

    def __init__(self, input_dim: int, ratio: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        original_num_nodes = x.size(0)
        perm = batched_random_topk(batch, self.ratio)
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = subgraph(
            perm,
            edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=original_num_nodes,
        )

        return x, edge_index, edge_attr, batch, perm, x.new_ones(x.size(0))


class SparsePooling(nn.Module):
    """GCN classifier with two sparse pooling stages."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model: str = "topk",
        hidden: int = 32,
        pratio: float = 0.5,
        dropout: float = 0.5,
    ):
        super().__init__()

        if model not in {"sag", "topk", "ndrp"}:
            raise ValueError("model must be one of 'sag', 'topk', or 'ndrp'")

        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        if model == "sag":
            pooling_layer = SAGPooling
        elif model == "topk":
            pooling_layer = TopKPooling
        else:
            pooling_layer = NDRPPooling
        self.pool1 = pooling_layer(hidden, ratio=pratio)
        self.pool2 = pooling_layer(hidden, ratio=pratio)

        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_classes)
        self.last_pool_time = 0.0

    def _timed_pool(self, pool, x, edge_index, batch):
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        pool_start = time.perf_counter()

        result = pool(x, edge_index, batch=batch)

        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        self.last_pool_time += time.perf_counter() - pool_start
        return result

    def forward(self, x, edge_index, batch):
        self.last_pool_time = 0.0

        # GCNConv: input_dim -> 32
        x = self.conv1(x, edge_index)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = 0.5
        x, edge_index, _, batch, _, _ = self._timed_pool(
            self.pool1, x, edge_index, batch
        )

        # GCNConv: 32 -> 32
        x = self.conv2(x, edge_index)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = 0.5
        x, edge_index, _, batch, _, _ = self._timed_pool(
            self.pool2, x, edge_index, batch
        )

        # GCNConv: 32 -> 32
        x = self.conv3(x, edge_index)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Readout: element-wise mean
        mean = global_mean_pool(x, batch)

        # Readout: element-wise max
        maximum = global_max_pool(x, batch)

        # 64-dimensional graph representation: concatenate 32-d mean and 32-d max
        graph_representation = torch.cat([mean, maximum], dim=1)

        # Linear output layer
        return self.classifier(graph_representation)


sparse_pooling = SparsePooling
