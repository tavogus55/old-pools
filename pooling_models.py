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
        selected_nodes = []
        for graph_id in batch.unique(sorted=True):
            graph_nodes = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            node_count = max(1, int(round(self.ratio * graph_nodes.numel())))
            selected = graph_nodes[
                torch.randperm(graph_nodes.numel(), device=x.device)[:node_count]
            ]
            selected_nodes.append(selected)

        perm = torch.cat(selected_nodes)
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

    def forward(self, x, edge_index, batch):
        # GCNConv: input_dim -> 32
        x = self.conv1(x, edge_index)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = 0.5
        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, batch=batch
        )

        # GCNConv: 32 -> 32
        x = self.conv2(x, edge_index)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = 0.5
        x, edge_index, _, batch, _, _ = self.pool2(
            x, edge_index, batch=batch
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
