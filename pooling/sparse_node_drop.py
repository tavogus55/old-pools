import torch
from torch_geometric.utils import to_dense_batch


def sparse_node_drop_pool(x, edge_index, batch, pr):
    """Uniform node dropping with a sparse induced-subgraph representation.

    For every graph, sample ``round(pr * n)`` distinct nodes uniformly without
    replacement, then retain exactly the edges whose endpoints were selected.
    This is the sparse equivalent of ``X[S], A[S, S]``.
    """
    num_nodes = x.size(0)
    node_ids = torch.arange(num_nodes, device=x.device)
    node_ids_dense, mask = to_dense_batch(node_ids, batch, fill_value=-1)

    random_scores = torch.rand(num_nodes, device=x.device)
    random_dense, _ = to_dense_batch(random_scores, batch, fill_value=-1.0)
    random_dense = random_dense.masked_fill(~mask, -1.0)

    nodes_per_graph = mask.sum(dim=1)
    keep_per_graph = torch.round(nodes_per_graph * pr).long().clamp(min=1)
    max_keep = int(keep_per_graph.max())

    positions = random_dense.topk(
        k=max_keep, dim=1, largest=True, sorted=False
    ).indices
    selected = node_ids_dense.gather(1, positions)
    selected_mask = (
        torch.arange(max_keep, device=x.device).unsqueeze(0)
        < keep_per_graph.unsqueeze(1)
    )
    perm = selected[selected_mask]

    keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
    keep_mask[perm] = True
    edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    pooled_edge_index = edge_index[:, edge_mask]

    new_index = torch.full((num_nodes,), -1, dtype=torch.long, device=x.device)
    new_index[perm] = torch.arange(perm.numel(), device=x.device)
    pooled_edge_index = new_index[pooled_edge_index]

    return x[perm], pooled_edge_index, batch[perm]
