"""Sparse structural pooling operators used by the shared experiment model."""

import torch
from torch_geometric.nn import graclus
from torch_geometric.utils import coalesce
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min


def _maximal_independent_set(edge_index, num_nodes):
    """Duan-pooling's vectorized canonical-order 1-MIS algorithm.

    The input may contain several disconnected graphs in one sparse batch.
    Since there are no cross-graph edges, one batched invocation gives the
    same MIS candidates as independent invocations, without Python loops.
    """
    if edge_index.numel() == 0:
        return torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)

    row, col = edge_index
    rank = torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
    mis = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    covered = mis.clone()
    min_rank = rank.clone()

    while not covered.all():
        min_neighbour_rank = torch.full_like(min_rank, fill_value=num_nodes)
        scatter_min(min_rank[row], col, out=min_neighbour_rank)
        torch.minimum(min_neighbour_rank, min_rank, out=min_rank)
        mis |= rank == min_rank

        covered = mis.to(torch.uint8)
        max_neighbour_covered = torch.zeros_like(covered)
        scatter_max(covered[row], col, out=max_neighbour_covered)
        torch.maximum(max_neighbour_covered, covered, out=covered)
        covered = covered.to(torch.bool)
        min_rank = rank.clone()
        min_rank[covered] = num_nodes

    return mis


def sparse_ndp_pool(x, edge_index, batch, ratio):
    """Pool each graph by vectorized maximal-independent-set decimation.

    This uses Duan-pooling's GPU-vectorized canonical-order MIS calculation.
    Candidate nodes are then capped independently per graph, rather than once
    for the whole mini-batch, so every graph retains at least one node.
    """
    num_nodes = x.size(0)
    num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    mis = _maximal_independent_set(edge_index, num_nodes)

    nodes_per_graph = torch.bincount(batch, minlength=num_graphs)
    keep_per_graph = (nodes_per_graph.to(torch.float) * ratio).long().clamp(min=1)
    candidates_per_graph = scatter_add(mis.long(), batch, dim=0, dim_size=num_graphs)
    candidate_offsets = torch.cat([
        candidates_per_graph.new_zeros(1), candidates_per_graph.cumsum(0)[:-1]
    ])
    candidate_rank = mis.long().cumsum(0) - 1 - candidate_offsets[batch]
    keep_mask = mis & (candidate_rank < keep_per_graph[batch])
    perm = keep_mask.nonzero(as_tuple=False).view(-1)

    source, target = edge_index
    node_map = torch.full((x.size(0),), -1, dtype=torch.long, device=x.device)
    node_map[perm] = torch.arange(perm.numel(), device=x.device)
    edge_mask = (node_map[source] >= 0) & (node_map[target] >= 0)
    pooled_edge_index = node_map[edge_index[:, edge_mask]]
    return x[perm], pooled_edge_index, batch[perm]


def sparse_graclus_pool(x, edge_index, batch):
    """Pool a sparse mini-batch with PyG's standard Graclus clustering."""
    if x.size(0) == 0 or edge_index.numel() == 0:
        return x, edge_index, batch

    cluster = graclus(edge_index, num_nodes=x.size(0))
    # ``graclus`` creates cluster IDs for the complete disconnected batch;
    # input graphs have no cross-graph edges, so clusters remain disjoint.
    x = scatter_mean(x, cluster, dim=0)
    pooled_batch = scatter_mean(batch.to(x.dtype), cluster, dim=0).long()
    pooled_edge_index = torch.stack([cluster[edge_index[0]], cluster[edge_index[1]]])
    pooled_edge_index = coalesce(
        pooled_edge_index, None, num_nodes=x.size(0), reduce='add'
    )
    return x, pooled_edge_index, pooled_batch
