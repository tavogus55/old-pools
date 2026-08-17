import torch
from torch_geometric.utils import coalesce


def sparse_countsketch_pool(x, edge_index, batch, pr, q, edge_weight=None):
    """Native sparse q-CountSketch computing SX and SAS^T.

    The sketch matrix is represented by ``q`` cluster/value pairs per node;
    it is never materialized. Each input edge expands to at most ``q^2``
    signed pooled-edge contributions, which are coalesced afterward.
    """
    if q < 1:
        raise ValueError(f'q must be positive, got {q}')

    device = x.device
    num_graphs = int(batch.max()) + 1 if batch.numel() else 0
    counts = torch.bincount(batch, minlength=num_graphs)
    clusters_per_graph = torch.clamp(
        (counts.to(torch.float) * pr).to(torch.long), min=1
    )
    offsets = torch.cat([
        clusters_per_graph.new_zeros(1),
        clusters_per_graph.cumsum(0),
    ])

    # Directly sample q distinct buckets per node in O(nq), rather than
    # allocating an n-by-p random matrix and selecting its top q entries.
    node_cluster_counts = clusters_per_graph[batch]
    node_offsets = offsets[:-1][batch]
    q_per_node = node_cluster_counts.clamp(max=q)
    clusters = torch.zeros((x.size(0), q), dtype=torch.long, device=device)
    valid = torch.zeros_like(clusters, dtype=torch.bool)
    for slot in range(q):
        slot_valid = slot < q_per_node
        valid[:, slot] = slot_valid
        available = (node_cluster_counts - slot).clamp_min(1)
        candidate = torch.floor(
            torch.rand(x.size(0), device=device) * available
        ).long()
        if slot:
            previous = torch.sort(clusters[:, :slot], dim=1).values
            for previous_bucket in previous.unbind(dim=1):
                candidate = candidate + (candidate >= previous_bucket).long()
        clusters[:, slot] = candidate

    scale = q_per_node.to(x.dtype).rsqrt().unsqueeze(1)
    values = (
        x.new_empty((x.size(0), q)).random_(2).mul_(2).sub_(1) * scale
    )
    values = values * valid.to(x.dtype)
    clusters = clusters + node_offsets.unsqueeze(1)
    q_width = q
    total_clusters = int(offsets[-1])

    assignment_nodes, assignment_slots = valid.nonzero(as_tuple=True)
    assignment_clusters = clusters[assignment_nodes, assignment_slots]
    assignment_values = values[assignment_nodes, assignment_slots]

    # X' = SX via sparse indexed accumulation.
    pooled_x = x.new_zeros((total_clusters, x.size(-1)))
    pooled_x.index_add_(
        0,
        assignment_clusters,
        x[assignment_nodes] * assignment_values.unsqueeze(-1),
    )

    if edge_weight is None:
        edge_weight = x.new_ones(edge_index.size(1))

    # A' = SAS^T. Each edge (u,v) contributes S[:,u] S[:,v]^T A_uv.
    source, target = edge_index
    pooled_source = clusters[source].unsqueeze(2).expand(-1, q_width, q_width)
    pooled_target = clusters[target].unsqueeze(1).expand(-1, q_width, q_width)
    pooled_edge_index = torch.stack([
        pooled_source.reshape(-1),
        pooled_target.reshape(-1),
    ])
    pooled_edge_weight = (
        values[source].unsqueeze(2)
        * values[target].unsqueeze(1)
        * edge_weight.view(-1, 1, 1)
    ).reshape(-1)
    pair_valid = (
        valid[source].unsqueeze(2) & valid[target].unsqueeze(1)
    ).reshape(-1)
    pooled_edge_index = pooled_edge_index[:, pair_valid]
    pooled_edge_weight = pooled_edge_weight[pair_valid]
    pooled_edge_index, pooled_edge_weight = coalesce(
        pooled_edge_index,
        pooled_edge_weight,
        num_nodes=total_clusters,
        reduce='sum',
    )

    pooled_batch = torch.repeat_interleave(
        torch.arange(num_graphs, device=device), clusters_per_graph
    )
    return pooled_x, pooled_edge_index, pooled_batch, pooled_edge_weight
