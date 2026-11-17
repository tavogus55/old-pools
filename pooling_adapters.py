"""Pipeline adapters for the vendored research implementations."""

import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_min
from torch_sparse import SparseTensor, coalesce

from pooling_originals.asym import AsymCheegerCutPool
from pooling_originals.pars import ParsingNet_GPU
from pooling_originals.sep import SEPooling


class AsymCheegerCutAdapter(AsymCheegerCutPool):
    """Original assignments and losses without a B*N*N*K broadcast."""

    def __init__(self, channels, num_clusters, compute_auxiliary_loss=True):
        super().__init__(
            num_clusters, mlp_channels=[channels, 128, 128],
            totvar_coeff=0.5, balance_coeff=0.5,
        )
        self.compute_auxiliary_loss = compute_auxiliary_loss

    def totvar_loss(self, adj, assignment):
        if not self.compute_auxiliary_loss:
            return assignment.new_zeros(assignment.size(0))
        batch_size, num_nodes, num_clusters = assignment.shape
        chunk_size = max(1, 8_000_000 // (num_nodes * num_nodes * num_clusters))
        losses = []
        for start in range(0, batch_size, chunk_size):
            selected = assignment[start:start + chunk_size]
            distances = torch.cdist(selected, selected, p=1)
            adjacency = adj[start:start + chunk_size]
            numerator = (adjacency * distances).sum(dim=(-1, -2))
            denominator = 2 * torch.count_nonzero(adjacency, dim=(-1, -2))
            losses.append(numerator / denominator.clamp_min(1))
        return torch.cat(losses)

    def balance_loss(self, assignment):
        if not self.compute_auxiliary_loss:
            return assignment.new_zeros(assignment.size(0))
        if self.k == 1:
            return assignment.sum(dim=(-1, -2)) * 0
        return super().balance_loss(assignment)


class ParsPooling(nn.Module):
    """Original neural parser and S.T @ X community-score/link-count gating."""

    def __init__(self, input_dim, ratio=0.5, dropout=0.2):
        super().__init__()
        self.parser = ParsingNet_GPU(
            channel=input_dim, dropout_network=dropout,
            dropout_parsing=dropout, layer_parsingnet=2,
            link_ignore_self_loop=True,
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        adjacency = SparseTensor.from_edge_index(
            edge_index, edge_attr, sparse_sizes=(x.size(0), x.size(0))
        )
        assignment, adjacency, pooled_batch, _, _, _, scores, counts = self.parser(
            x, adjacency, batch
        )
        pooled_x = assignment.t() @ x
        pooled_x = pooled_x * scores.view(-1, 1) * counts.view(-1, 1)
        row, col, _ = adjacency.coo()
        return pooled_x, torch.stack([row, col]), None, pooled_batch, None, scores


class SEPData(Data):
    """Offset both levels' cluster assignments correctly during PyG batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if key == "sep_assignment1":
            return self.sep_count1
        if key == "sep_assignment2":
            return self.sep_count2
        return super().__inc__(key, value, *args, **kwargs)


def prepare_sep_dataset(dataset, cache_path=None):
    """Build the original depth-three SEP-G tree once per unique topology."""
    import networkx as nx
    from pooling_originals.sep_transform import pool_trans_discon

    cache_path = Path(cache_path) if cache_path is not None else None
    cache = {}
    if cache_path is not None and cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu")
    prepared = []
    changed = False
    for data in dataset:
        if data.num_nodes < 1:
            raise ValueError("SEP requires graphs with at least one node")
        graph = nx.Graph()
        graph.add_nodes_from(range(data.num_nodes))
        graph.add_edges_from(data.edge_index.t().tolist())
        graph.remove_edges_from(nx.selfloop_edges(graph))
        edges = torch.tensor(
            sorted((min(source, target), max(source, target))
                   for source, target in graph.edges),
            dtype=torch.long,
        )
        digest = hashlib.sha256(
            str(data.num_nodes).encode() + edges.numpy().tobytes()
        ).hexdigest()
        if digest not in cache:
            if data.num_nodes == 1:
                assignments = (
                    torch.zeros(1, dtype=torch.long),
                    torch.zeros(1, dtype=torch.long),
                )
            else:
                tree = pool_trans_discon(({"G": graph}, 3))["tree"]
                assignments = []
                children = list(range(data.num_nodes))
                for depth in (1, 2):
                    parents = sorted(
                        node_id for node_id, node in tree.items()
                        if node["depth"] == depth
                    )
                    parent_index = {
                        node_id: index for index, node_id in enumerate(parents)
                    }
                    assignments.append(torch.tensor(
                        [parent_index[tree[node_id]["parent"]]
                         for node_id in children], dtype=torch.long,
                    ))
                    children = parents
                assignments = tuple(assignments)
            cache[digest] = assignments
            changed = True
        first, second = cache[digest]
        graph_data = SEPData.from_dict(data.to_dict())
        graph_data.sep_assignment1 = first.clone()
        graph_data.sep_assignment2 = second.clone()
        graph_data.sep_count1 = second.numel()
        graph_data.sep_count2 = int(second.max()) + 1
        prepared.append(graph_data)
    if cache_path is not None and changed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(cache, temporary_path)
        temporary_path.replace(cache_path)
    return prepared


class SEPPooling(nn.Module):
    """Original SEP-G child-to-parent sum and unweighted quotient graph."""

    def __init__(self, input_dim, ratio=0.5):
        super().__init__()
        self.pool = SEPooling(nn.Identity())

    def forward(self, x, edge_index, edge_attr=None, batch=None, assignment=None):
        if assignment is None or assignment.numel() != x.size(0):
            raise ValueError("SEP requires precomputed coding-tree assignments")
        if batch is None:
            batch = assignment.new_zeros(x.size(0))
        num_clusters = int(assignment.max()) + 1
        children = torch.arange(x.size(0), device=x.device)
        pooled_x = self.pool(
            (None, x), torch.stack([assignment, children]),
            size=(num_clusters, x.size(0)),
        )
        pooled_batch = scatter_min(batch, assignment, dim=0, dim_size=num_clusters)[0]
        pooled_edges, _ = remove_self_loops(assignment[edge_index])
        pooled_edges, _ = coalesce(pooled_edges, None, num_clusters, num_clusters)
        return pooled_x, pooled_edges, None, pooled_batch, None, None
