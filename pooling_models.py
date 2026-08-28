import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import (
    DenseGCNConv,
    DMoNPooling,
    GCNConv,
    GraphConv,
    ASAPooling,
    PANPooling,
    SAGPooling,
    TopKPooling,
    global_max_pool,
    global_mean_pool,
    dense_diff_pool,
    dense_mincut_pool,
)
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.pool import graclus
from torch_geometric.utils import (
    add_remaining_self_loops,
    add_self_loops,
    dense_to_sparse,
    remove_self_loops as torch_geometric_remove_self_loops,
    subgraph,
    softmax,
    to_dense_adj,
    to_dense_batch,
)
from torch_scatter import scatter, scatter_add, scatter_max, scatter_min
from torch_sparse import SparseTensor, coalesce, transpose


def make_message_passing_layer(mp_layer, input_dim, output_dim):
    """Create the selected sparse message-passing layer."""
    if mp_layer == "gcn":
        return GCNConv(input_dim, output_dim)
    if mp_layer == "graphconv":
        return GraphConv(input_dim, output_dim)
    raise ValueError("mp_layer must be 'gcn' or 'graphconv'")


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


def sample_countsketch_rows(p_node, q):
    """Sample distinct local CountSketch buckets without an N-by-P tensor."""
    q_node = p_node.clamp(max=q)
    n = p_node.numel()
    rows = torch.zeros((n, q), dtype=torch.long, device=p_node.device)
    valid = torch.arange(q, device=p_node.device)[None, :] < q_node[:, None]

    # Vectorized Floyd sampling; q is only 1, 2, or 4 in this experiment.
    for k in range(q):
        upper = p_node - q_node + k
        candidate = (torch.rand(n, device=p_node.device) * (upper + 1)).long()
        if k:
            duplicate = (candidate[:, None] == rows[:, :k]).any(dim=1)
            candidate = torch.where(duplicate, upper, candidate)
        rows[:, k] = torch.where(valid[:, k], candidate, 0)

    return rows, valid, q_node


class _ParsingMLP(nn.Module):
    """Small edge-scoring MLP used by the Graph Parsing Network parser."""

    def __init__(self, channels, num_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        if num_layers <= 1:
            layers.append(nn.Linear(channels, 1))
        else:
            layers.append(nn.Linear(channels, channels))
            for _ in range(num_layers - 2):
                layers.extend([nn.LayerNorm(channels), nn.ReLU(), nn.Dropout(dropout)])
                layers.append(nn.Linear(channels, channels))
            layers.extend([nn.LayerNorm(channels), nn.ReLU(), nn.Dropout(dropout)])
            layers.append(nn.Linear(channels, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ParsPooling(nn.Module):
    """Graph Parsing Network coarsening adapted to the sparse old-pools API.

    The paper parser learns edge scores, builds dominant-node communities, and
    returns a sparse node-to-community assignment.  This adapter uses that
    assignment to compute ``S.T @ X`` and contracts the sparse edge list.
    The parser determines the number of communities from graph structure, so
    ``ratio`` is accepted for interface compatibility but is not used by the
    original parsing algorithm.
    """

    def __init__(self, input_dim, ratio=0.5, dropout=0.2):
        super().__init__()
        self.ratio = ratio
        self.dropout_parsing = dropout
        self.edge_net = _ParsingMLP(input_dim, num_layers=2, dropout=dropout)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        num_nodes = x.size(0)
        device = x.device

        # Remove self-loops before parsing, matching GraphParsingNetworks.
        edge_index, edge_attr = torch_geometric_remove_self_loops(
            edge_index, edge_attr
        )
        if edge_index.numel() == 0:
            perm = torch.arange(num_nodes, device=device)
            return x, edge_index, edge_attr, batch, perm, x.new_ones(num_nodes)

        # Randomly drop parsing edges only while training, as in the paper code.
        if self.training and self.dropout_parsing > 0:
            keep = torch.rand(edge_index.size(1), device=device) >= self.dropout_parsing
            edge_index = edge_index[:, keep]
            if edge_attr is not None:
                edge_attr = edge_attr[keep]
            if edge_index.numel() == 0:
                perm = torch.arange(num_nodes, device=device)
                return x, edge_index, edge_attr, batch, perm, x.new_ones(num_nodes)

        row, col = edge_index
        edge_features = x[row] * x[col]
        edge_score = torch.sigmoid(self.edge_net(edge_features).view(-1))

        # The paper's parser is deliberately performed on CPU because its
        # dominant-edge/community loop uses stable sorting and set operations.
        cpu_edge = edge_index.detach().cpu()
        cpu_score = edge_score.detach().cpu()
        cpu_batch = batch.detach().cpu()
        cpu_row, cpu_col = cpu_edge
        # PyTorch 1.11 has no stable= argument on torch.argsort.
        order = torch.sort(cpu_score, descending=True)[1]
        sorted_edges = cpu_edge[:, order]
        sorted_scores = cpu_score[order]

        # Select each node's highest-scoring incident edge.
        best_rank = torch.full((num_nodes,), cpu_edge.size(1), dtype=torch.long)
        for rank, source in enumerate(sorted_edges[0].tolist()):
            if best_rank[source] == cpu_edge.size(1):
                best_rank[source] = rank
        connected = best_rank < cpu_edge.size(1)
        connected_nodes = connected.nonzero(as_tuple=False).view(-1)
        isolated_nodes = (~connected).nonzero(as_tuple=False).view(-1)
        dominant_edges = sorted_edges[:, best_rank[connected_nodes]]
        dominant_scores = sorted_scores[best_rank[connected_nodes]]
        # Use the PyTorch 1.11-compatible sorting form here as well.
        dominant_order = torch.sort(dominant_scores, descending=True)[1]
        dominant_edges = dominant_edges[:, dominant_order]

        node_to_comm = torch.full((num_nodes,), -1, dtype=torch.long)
        next_comm = 0
        remaining = dominant_edges
        # ``dominant_edges`` still contains global node IDs, so use the full
        # global batch vector rather than a connected-node-local batch vector.
        connected_batch = cpu_batch

        # Reproduce the paper's iterative dominant-edge parsing logic.
        while remaining.size(1):
            first_by_graph = []
            for graph_id in connected_batch[remaining[0]].unique().tolist():
                candidates = (connected_batch[remaining[0]] == graph_id).nonzero().view(-1)
                first_by_graph.append(candidates[0])
            node_set = remaining[:, torch.stack(first_by_graph)].unique()
            while True:
                subset_mask = torch.isin(remaining[1], node_set)
                subset = torch.cat([remaining[0][subset_mask], node_set]).unique()
                graph_ids = connected_batch[subset].unique()
                graph_comm = torch.full((int(cpu_batch.max().item()) + 1,), -1, dtype=torch.long)
                graph_comm[graph_ids] = torch.arange(graph_ids.numel()) + next_comm
                remaining = remaining[:, ~subset_mask]
                node_to_comm[subset] = graph_comm[connected_batch[subset]]
                if subset.numel() <= node_set.numel():
                    next_comm = int(node_to_comm.max().item()) + 1
                    break
                node_set = subset

        # Isolated nodes remain singleton communities.
        node_to_comm[isolated_nodes] = next_comm + torch.arange(isolated_nodes.numel())
        num_communities = int(node_to_comm.max().item()) + 1

        node_to_comm = node_to_comm.to(device)
        assignment = SparseTensor(
            row=torch.arange(num_nodes, device=device),
            col=node_to_comm,
            sparse_sizes=(num_nodes, num_communities),
        )

        # Pool features through the same sparse assignment matrix S.T @ X.
        pooled_x = assignment.t() @ x
        pooled_batch = scatter_min(batch, node_to_comm, dim=0)[0]

        # Contract the original sparse graph and merge duplicate community edges.
        pooled_edges = torch.stack(
            [node_to_comm[edge_index[0]], node_to_comm[edge_index[1]]], dim=0
        )
        pooled_adj = torch.sparse_coo_tensor(
            pooled_edges,
            x.new_ones(pooled_edges.size(1)),
            (num_communities, num_communities),
            device=device,
        ).coalesce()
        pooled_edge_index = pooled_adj.indices()
        pooled_edge_weight = pooled_adj.values()
        return (
            pooled_x,
            pooled_edge_index,
            pooled_edge_weight,
            pooled_batch,
            None,
            x.new_ones(num_communities),
        )


def countsketch_pool(
    x, edge_index, pr, q, batch=None, edge_weight=None
):
    """Sparse CountSketch pooling for a PyG graph or mini-batch.

    This implements the same mathematical operations as the original dense
    version, X_pool = S X and A_pool = S A S.T, but never materializes S.
    Only the non-zero assignment locations are sampled and stored.
    """
    if x.dim() != 2 or edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError("x must be [N,d] and edge_index must be [2,E]")
    if not 0.0 < pr <= 1.0 or not isinstance(q, int) or q < 1:
        raise ValueError("require 0 < pr <= 1 and integer q >= 1")

    n, feature_dim = x.shape
    if n == 0:
        raise ValueError("x must contain at least one node")
    if batch is None:
        batch = torch.zeros(n, dtype=torch.long, device=x.device)
    if batch.numel() != n:
        raise ValueError("batch must contain one graph id per node")

    num_graphs = int(batch.max().item()) + 1
    num_nodes_per_graph = torch.bincount(batch, minlength=num_graphs)
    num_clusters_per_graph = (num_nodes_per_graph * pr).long().clamp_min(1)
    pool_ptr = torch.cat(
        (num_clusters_per_graph.new_zeros(1), num_clusters_per_graph.cumsum(0))
    )
    total_clusters = int(pool_ptr[-1].item())

    # Sample graph-local buckets and offset them into the batched cluster space.
    p_node = num_clusters_per_graph[batch]
    rows, valid, q_node = sample_countsketch_rows(p_node, q)
    rows = rows + pool_ptr[batch][:, None]

    # Generate the same scaled Rademacher signs used by CountSketch.
    signs = x.new_empty((n, q)).random_(2).mul_(2).sub_(1)
    signs *= valid / q_node.to(x.dtype).sqrt()[:, None]

    # Original operation: X_pool = S X, using only non-zero assignments.
    pooled_x = x.new_zeros((total_clusters, feature_dim))
    for assignment in range(q):
        keep = valid[:, assignment]
        pooled_x = pooled_x.index_add(
            0,
            rows[keep, assignment],
            signs[keep, assignment, None] * x[keep],
        )

    src, dst = edge_index
    if edge_weight is None:
        edge_weight = x.new_ones(src.numel())
    if edge_weight.dim() != 1 or edge_weight.numel() != src.numel():
        raise ValueError("edge_weight must be a scalar vector of length E")

    # Original operation: A_pool = S A S.T. Each edge produces at most q^2
    # pooled-edge events, which are combined by sparse COO coalescing.
    pair = valid[src, :, None] & valid[dst, None, :]
    out_src = rows[src, :, None].expand(-1, q, q)[pair]
    out_dst = rows[dst, None, :].expand(-1, q, q)[pair]
    out_weight = (
        edge_weight[:, None, None]
        * signs[src, :, None]
        * signs[dst, None, :]
    )[pair]

    pooled_adjacency = torch.sparse_coo_tensor(
        torch.stack((out_src, out_dst)),
        out_weight,
        (total_clusters, total_clusters),
        device=x.device,
    ).coalesce()
    nonzero = pooled_adjacency.values() != 0
    pooled_edge_index = pooled_adjacency.indices()[:, nonzero]
    pooled_edge_weight = pooled_adjacency.values()[nonzero]
    pooled_batch = torch.repeat_interleave(
        torch.arange(num_graphs, device=x.device), num_clusters_per_graph
    )

    return pooled_x, pooled_edge_index, pooled_edge_weight, pooled_batch


def maximal_independent_set(edge_index, num_nodes=None):
    """Return a greedy maximal independent set as a boolean node mask."""
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0

    row, col = edge_index
    rank = torch.arange(num_nodes, device=edge_index.device)
    mis = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    active = mis.clone()
    min_rank = rank.clone()

    while not active.all():
        minimum = torch.full_like(min_rank, num_nodes)
        scatter_min(min_rank[row], col, out=minimum)
        torch.minimum(minimum, min_rank, out=min_rank)
        mis |= rank == min_rank

        active = mis.clone()
        maximum = torch.zeros(
            num_nodes,
            dtype=torch.long,
            device=edge_index.device,
        )
        scatter_max(active[row].to(torch.long), col, out=maximum)
        active |= maximum.bool()

        min_rank = rank.clone()
        min_rank[active] = num_nodes

    return mis


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


class NDPPooling(nn.Module):
    """Decimation pooling using a maximal independent set."""

    def __init__(self, input_dim: int, ratio: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        num_nodes = x.size(0)
        if edge_index.numel() == 0:
            perm = torch.arange(num_nodes, device=x.device)
            return x, edge_index, edge_attr, batch, perm, x.new_ones(num_nodes)

        mis = maximal_independent_set(edge_index, num_nodes=num_nodes)
        graph_count = int(batch.max().item()) + 1
        graph_counts = scatter_add(
            batch.new_ones(num_nodes), batch, dim=0, dim_size=graph_count
        )
        graph_starts = torch.cat([
            graph_counts.new_zeros(1),
            graph_counts.cumsum(dim=0)[:-1],
        ])
        graph_keep = (
            graph_counts.to(torch.float) * self.ratio
        ).long().clamp_min(1)

        selected = mis.nonzero(as_tuple=False).view(-1)
        selected_batch = batch[selected]
        selected_counts = scatter_add(
            selected_batch.new_ones(selected_batch.numel()),
            selected_batch,
            dim=0,
            dim_size=graph_count,
        )
        selected_offsets = selected_counts.cumsum(0) - selected_counts
        selected_rank = (
            torch.arange(selected.numel(), device=x.device)
            - torch.repeat_interleave(selected_offsets, selected_counts)
        )
        perm = selected[
            selected_rank < graph_keep[selected_batch]
        ]

        missing = (selected_counts == 0).nonzero(as_tuple=False).view(-1)
        if missing.numel() > 0:
            missing_counts = graph_keep[missing]
            missing_offsets = missing_counts.cumsum(0) - missing_counts
            missing_rank = (
                torch.arange(missing_counts.sum(), device=x.device)
                - torch.repeat_interleave(missing_offsets, missing_counts)
            )
            fallback = (
                graph_starts[missing].repeat_interleave(missing_counts)
                + missing_rank
            )
            perm = torch.cat([perm, fallback])

        perm = perm[torch.argsort(batch[perm])]
        x = x[perm]
        batch = batch[perm]
        edge_index, edge_attr = subgraph(
            perm,
            edge_index,
            edge_attr=edge_attr,
            relabel_nodes=True,
            num_nodes=num_nodes,
        )
        return x, edge_index, edge_attr, batch, perm, x.new_ones(x.size(0))


class GraclusPooling(nn.Module):
    """Sparse Graclus clustering followed by mean feature aggregation."""

    def __init__(self, input_dim: int, ratio: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.ratio = ratio

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        # Graclus creates a cluster assignment from the sparse graph structure.
        cluster = graclus(edge_index, num_nodes=x.size(0))
        _, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        num_clusters = int(cluster.max().item()) + 1 if cluster.numel() else 0

        # Mean-pool node features within each Graclus cluster.
        pooled_x = x.new_zeros((num_clusters, x.size(1)))
        pooled_x.index_add_(0, cluster, x)
        cluster_size = scatter_add(
            x.new_ones(x.size(0)), cluster, dim=0, dim_size=num_clusters
        )
        pooled_x = pooled_x / cluster_size.clamp_min(1).unsqueeze(-1)

        # Contract sparse edges according to the cluster assignment and merge
        # duplicate cluster pairs into one sparse edge.
        if edge_index.numel():
            pooled_edges = torch.stack(
                [cluster[edge_index[0]], cluster[edge_index[1]]], dim=0
            )
            pooled_adjacency = torch.sparse_coo_tensor(
                pooled_edges,
                x.new_ones(pooled_edges.size(1)),
                (num_clusters, num_clusters),
                device=x.device,
            ).coalesce()
            pooled_edge_index = pooled_adjacency.indices()
        else:
            pooled_edge_index = edge_index.new_empty((2, 0))

        pooled_batch = scatter_min(batch, cluster, dim=0)[0]
        return (
            pooled_x,
            pooled_edge_index,
            None,
            pooled_batch,
            None,
            x.new_ones(num_clusters),
        )


class ASAPoolingAdapter(nn.Module):
    """Adapt PyG ASAPooling's five-value output to the sparse model interface."""

    def __init__(self, input_dim: int, ratio: float, mp_layer: str):
        super().__init__()
        message_passing = GCNConv if mp_layer == "gcn" else GraphConv
        self.pool = ASAPooling(
            input_dim,
            ratio=ratio,
            GNN=message_passing,
        )

    def forward(self, x, edge_index, batch=None):
        x, edge_index, edge_weight, batch, perm = self.pool(
            x, edge_index, batch=batch
        )
        return x, edge_index, edge_weight, batch, perm, x.new_ones(x.size(0))


class PANPoolingAdapter(nn.Module):
    """Adapt PANPooling's SparseTensor input to the sparse model interface."""

    def __init__(self, input_dim: int, ratio: float):
        super().__init__()
        self.pool = PANPooling(input_dim, ratio=ratio)

    def forward(self, x, edge_index, batch=None):
        # PANPooling expects a SparseTensor with explicit unit edge weights.
        edge_weight = x.new_ones(edge_index.size(1))
        adjacency = SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=edge_weight,
            sparse_sizes=(x.size(0), x.size(0)),
        )
        return self.pool(x, adjacency, batch=batch)


class CoPoolingGraphAttention(nn.Module):
    """Graph-attention scorer copied from the benchmark-paper CoPooling code."""

    def __init__(self, num_in_features, num_out_features, num_of_heads=1):
        super().__init__()
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.linear_proj = nn.Linear(
            num_in_features, num_of_heads * num_out_features, bias=False
        )
        self.scoring_fn_target = Parameter(
            torch.Tensor(1, num_of_heads, num_out_features)
        )
        self.scoring_fn_source = Parameter(
            torch.Tensor(1, num_of_heads, num_out_features)
        )
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def forward(self, x, edge_index):
        projected = self.linear_proj(x).view(
            -1, self.num_of_heads, self.num_out_features
        )
        source = (projected * self.scoring_fn_source).sum(dim=-1)
        target = (projected * self.scoring_fn_target).sum(dim=-1)
        source = source.index_select(0, edge_index[0])
        target = target.index_select(0, edge_index[1])
        return torch.sigmoid(source + target)


class CoPoolingGPRProp(MessagePassing):
    """GPR propagation copied from the benchmark-paper CoPooling code."""

    def __init__(self, K, alpha, Init, Gamma=None):
        super().__init__(aggr="add")
        self.K = K
        self.alpha = alpha
        if Init == "Random":
            bound = np.sqrt(3 / (K + 1))
            temp = np.random.uniform(-bound, bound, K + 1)
            temp = temp / np.sum(np.abs(temp))
        elif Init == "PPR":
            temp = alpha * (1 - alpha) ** np.arange(K + 1)
            temp[-1] = (1 - alpha) ** K
        elif Init == "NPPR":
            temp = alpha ** np.arange(K + 1)
            temp = temp / np.sum(np.abs(temp))
        elif Init == "WS":
            temp = Gamma
        elif Init == "SGC":
            temp = np.zeros(K + 1)
            temp[-1] = 1.0
        else:
            raise ValueError("Unsupported CoPooling initialization")
        self.temp = Parameter(torch.tensor(temp, dtype=torch.float))

    def reset_parameters(self):
        with torch.no_grad():
            self.temp.zero_()
            for k in range(self.K + 1):
                self.temp[k] = self.alpha * (1 - self.alpha) ** k
            self.temp[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype
        )
        hidden = x * self.temp[0]
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            hidden = hidden + self.temp[k + 1] * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class CoPoolingInformationScore(MessagePassing):
    """Node-information score copied from the benchmark-paper code."""

    def __init__(self):
        super().__init__(aggr="add")

    def forward(self, x, edge_index, edge_weight):
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 0, x.size(0)
        )
        row, col = edge_index
        degree = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        degree_inv_sqrt = degree.pow(-0.5)
        degree_inv_sqrt[degree_inv_sqrt == float("inf")] = 0
        expanded_degree = x.new_zeros(edge_weight.size(0))
        expanded_degree[-x.size(0):] = 1
        norm = expanded_degree - (
            degree_inv_sqrt[row] * edge_weight * degree_inv_sqrt[col]
        )
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


def copooling_cumsum(x):
    return torch.cat([x.new_zeros(1), x.cumsum(dim=0)])


def copooling_topk(x, ratio, batch):
    """Paper top-k selection, adapted to the current batched pipeline."""
    num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce="sum")
    k = (float(ratio) * num_nodes.to(x.dtype)).ceil().long().clamp_min(1)
    x, x_perm = torch.sort(x.view(-1), descending=True)
    sorted_batch = batch[x_perm]
    sorted_batch, batch_perm = torch.sort(
        sorted_batch, descending=False, stable=True
    )
    arange = torch.arange(x.size(0), device=x.device)
    positions = arange - copooling_cumsum(num_nodes)[sorted_batch]
    return x_perm[batch_perm[positions < k[sorted_batch]]]


def copooling_filter_adj(edge_index, edge_attr, perm, num_nodes):
    mapping = perm.new_full((num_nodes,), -1)
    mapping[perm] = torch.arange(perm.numel(), device=perm.device)
    row, col = edge_index
    row, col = mapping[row], mapping[col]
    keep = (row >= 0) & (col >= 0)
    if edge_attr is not None:
        edge_attr = edge_attr[keep]
    return torch.stack([row[keep], col[keep]]), edge_attr


class CoPoolingAdapter(nn.Module):
    """Paper CoPooling adapted to old-pools' common sparse-pool interface."""

    def __init__(self, input_dim, ratio=0.5, mp_layer="graphconv"):
        super().__init__()
        self.ratio = ratio
        self.edge_ratio = 0.6
        # Researcher-requested COP default: propagation depth k=2.
        self.prop = CoPoolingGPRProp(2, 0.1, "Random", 1.0)
        self.graph_attention = CoPoolingGraphAttention(input_dim, 32, 1)
        self.information_score = CoPoolingInformationScore()
        self.weight = Parameter(torch.Tensor(2 * input_dim, input_dim))
        self.bias = Parameter(torch.Tensor(input_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.prop.reset_parameters()
        self.graph_attention.init_params()

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        num_nodes = x.size(0)

        # Paper step: propagate node features before calculating edge attention.
        x_cut = self.prop(x, edge_index)

        # Paper step: calculate one attention value for every input edge.
        attention = self.graph_attention(x_cut, edge_index).sum(dim=1)

        # Paper step: add self-loops, symmetrize, and average duplicate edges.
        edge_index, attention = add_self_loops(
            edge_index, attention, 1.0, num_nodes
        )
        reverse_edge_index, reverse_attention = transpose(
            edge_index, attention, num_nodes, num_nodes
        )
        edge_index, attention = coalesce(
            torch.cat([edge_index, reverse_edge_index], dim=1),
            torch.cat([attention, reverse_attention]),
            num_nodes,
            num_nodes,
            "mean",
        )

        # Paper step: retain the top edge_ratio percentile using NumPy percentile.
        cut_value = np.percentile(
            attention.detach().cpu().numpy(),
            int(100 * (1 - self.edge_ratio)),
        )
        attention = attention * (attention >= cut_value)
        keep_edges = attention > 0
        cut_edge_index = edge_index[:, keep_edges]
        cut_edge_attr = attention[keep_edges]

        # Paper step: score nodes and select the pooling ratio per graph.
        node_score = self.information_score(
            x, cut_edge_index, cut_edge_attr
        ).abs().sum(dim=1)
        perm = copooling_topk(node_score, self.ratio, batch)
        pooled_x = x[perm]
        pooled_batch = batch[perm]
        pooled_edge_index, pooled_edge_attr = copooling_filter_adj(
            cut_edge_index, cut_edge_attr, perm, num_nodes
        )

        # Paper step: apply the attention-based feature update after pooling.
        attention_dense = to_dense_adj(
            cut_edge_index,
            edge_attr=cut_edge_attr,
            max_num_nodes=num_nodes,
        ).squeeze(0)
        pooled_x = F.relu(
            torch.cat([pooled_x, attention_dense[perm] @ x], dim=1)
            @ self.weight
            + self.bias
        )
        return (
            pooled_x,
            pooled_edge_index,
            pooled_edge_attr,
            pooled_batch,
            perm,
            pooled_x.new_ones(pooled_x.size(0)),
        )


class CGIPool(nn.Module):
    """CGI pooling from the benchmark paper, adapted to six outputs."""

    def __init__(self, in_channels, ratio=0.5, non_lin=torch.tanh):
        super().__init__()
        self.ratio = ratio
        self.non_lin = non_lin
        self.transform = GraphConv(in_channels, in_channels)
        self.pp_conv = GraphConv(in_channels, in_channels)
        self.np_conv = GraphConv(in_channels, in_channels)
        self.positive_pooling = GraphConv(in_channels, 1)
        self.negative_pooling = GraphConv(in_channels, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        x_transform = F.leaky_relu(self.transform(x, edge_index), 0.2)
        x_tp = F.leaky_relu(self.pp_conv(x, edge_index), 0.2)
        x_tn = F.leaky_relu(self.np_conv(x, edge_index), 0.2)
        score_positive = self.positive_pooling(x_tp, edge_index).view(-1)
        score_negative = self.negative_pooling(x_tn, edge_index).view(-1)
        positive = copooling_topk(score_positive, 1, batch)
        negative = copooling_topk(score_negative, 1, batch)
        # The paper uses positive/negative graph readouts for CGI's auxiliary
        # discriminator; the classifier path uses their score difference.
        score = score_positive - score_negative
        perm = copooling_topk(score, self.ratio, batch)
        pooled_x = x_transform[perm] * self.non_lin(score[perm]).view(-1, 1)
        pooled_edge_index, pooled_edge_attr = subgraph(
            perm,
            edge_index,
            relabel_nodes=True,
            num_nodes=x.size(0),
        )
        return pooled_x, pooled_edge_index, pooled_edge_attr, batch[perm], perm, pooled_x.new_ones(pooled_x.size(0))


def kmis_cluster(edge_index, k=1, permutation=None, num_nodes=None):
    """Benchmark-paper k-MIS clustering for sparse COO graphs."""
    if num_nodes is None:
        num_nodes = int(edge_index.max().item()) + 1 if edge_index.numel() else 0
    row, col = edge_index
    if permutation is None:
        rank = torch.arange(num_nodes, device=edge_index.device)
    else:
        rank = torch.zeros_like(permutation)
        rank[permutation] = torch.arange(num_nodes, device=edge_index.device)
    mis = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
    active = mis.clone()
    while not active.all():
        min_rank = rank.clone()
        min_rank[active] = num_nodes
        for _ in range(k):
            minimum = torch.full_like(min_rank, num_nodes)
            scatter_min(min_rank[row], col, out=minimum)
            torch.minimum(minimum, min_rank, out=min_rank)
        mis |= rank == min_rank
        active = mis.clone()
        for _ in range(k):
            maximum = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
            scatter_max(active[row].long(), col, out=maximum)
            active |= maximum.bool()
    min_rank = torch.full((num_nodes,), num_nodes, dtype=torch.long, device=edge_index.device)
    min_rank[mis] = rank[mis]
    for _ in range(k):
        minimum = torch.full_like(min_rank, num_nodes)
        scatter_min(min_rank[row], col, out=minimum)
        torch.minimum(minimum, min_rank, out=min_rank)
    _, cluster = torch.unique(min_rank, return_inverse=True)
    mis_indices = mis.nonzero(as_tuple=False).view(-1)
    mis_order = torch.argsort(rank[mis])
    return mis, mis_order[cluster]


class KMISPool(nn.Module):
    """KMIS pooling from the benchmark paper, with its linear scorer."""

    def __init__(self, in_channels, k=3):
        super().__init__()
        self.k = k
        self.lin = nn.Linear(in_channels, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        score = self.lin(x).sigmoid().view(-1)
        updated_score = score.clone()
        row, col = edge_index
        for _ in range(self.k):
            scatter_add(updated_score[row], col, out=updated_score)
        updated_score = score / updated_score
        permutation = torch.argsort(updated_score, descending=True)
        mis, cluster = kmis_cluster(edge_index, self.k, permutation, x.size(0))
        num_clusters = int(mis.sum().item())
        pooled_edges = torch.stack([cluster[row], cluster[col]], dim=0)
        pooled_adjacency = torch.sparse_coo_tensor(
            pooled_edges,
            x.new_ones(pooled_edges.size(1)),
            (num_clusters, num_clusters),
        ).coalesce()
        pooled_edges = pooled_adjacency.indices()
        keep = pooled_edges[0] != pooled_edges[1]
        pooled_edges = pooled_edges[:, keep]
        pooled_x = x[mis] * score[mis].view(-1, 1)
        pooled_batch = batch[mis]
        return pooled_x, pooled_edges, None, pooled_batch, mis, cluster


class GSAPool(nn.Module):
    """GSAPool from the benchmark paper, adapted to the shared interface."""

    def __init__(self, in_channels, pooling_ratio=0.5, alpha=0.4):
        super().__init__()
        self.ratio = pooling_ratio
        self.alpha = alpha
        self.sbtl_layer = GCNConv(in_channels, 1)
        self.fbtl_layer = nn.Linear(in_channels, 1)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        score_s = self.sbtl_layer(x, edge_index).view(-1)
        score_f = self.fbtl_layer(x).view(-1)
        score = torch.tanh(score_s * self.alpha + score_f * (1 - self.alpha))
        perm = copooling_topk(score, self.ratio, batch)
        pooled_x = x[perm] * score[perm].view(-1, 1)
        pooled_edge_index, pooled_edge_attr = subgraph(
            perm, edge_index, relabel_nodes=True, num_nodes=x.size(0)
        )
        return pooled_x, pooled_edge_index, pooled_edge_attr, batch[perm], perm, x[perm]


class HGPSLPool(nn.Module):
    """HGPSL pooling from the benchmark paper (default non-sampled path)."""

    def __init__(self, in_channels, ratio=0.5, lamb=1.0, negative_slop=0.2):
        super().__init__()
        self.ratio = ratio
        self.lamb = lamb
        self.negative_slop = negative_slop
        self.att = Parameter(torch.Tensor(1, 2 * in_channels))
        nn.init.xavier_uniform_(self.att)
        self.information_score = CoPoolingInformationScore()

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        score = self.information_score(
            x, edge_index, x.new_ones(edge_index.size(1))
        ).abs().sum(dim=1)
        perm = copooling_topk(score, self.ratio, batch)
        pooled_x = x[perm]
        pooled_batch = batch[perm]
        induced_edge_index, induced_edge_attr = subgraph(
            perm, edge_index, relabel_nodes=True, num_nodes=x.size(0)
        )

        # Paper step: construct the learned link structure and normalize it.
        num_nodes = pooled_x.size(0)
        row, col = induced_edge_index
        if induced_edge_attr is None:
            induced_edge_attr = pooled_x.new_ones(row.size(0))
        graph_counts = scatter_add(
            pooled_batch.new_ones(num_nodes), pooled_batch, dim=0
        )
        graph_starts = torch.cat(
            [graph_counts.new_zeros(1), graph_counts.cumsum(0)[:-1]]
        )
        # HGPSL's default benchmark path creates a complete block per graph.
        dense_adj = pooled_x.new_zeros((num_nodes, num_nodes))
        for start, end in zip(graph_starts, graph_counts.cumsum(0)):
            dense_adj[start:end, start:end] = 1.0
        full_edge_index, _ = dense_to_sparse(dense_adj)
        full_row, full_col = full_edge_index
        weights = (
            torch.cat([pooled_x[full_row], pooled_x[full_col]], dim=1) * self.att
        ).sum(dim=-1)
        weights = F.leaky_relu(weights, self.negative_slop)
        dense_adj[full_row, full_col] = weights
        induced_row, induced_col = induced_edge_index
        dense_adj[induced_row, induced_col] += induced_edge_attr * self.lamb
        weights = dense_adj[full_row, full_col]
        normalized = softmax(weights, full_row, num_nodes=num_nodes)
        return pooled_x, full_edge_index, normalized, pooled_batch, perm, pooled_x.new_ones(num_nodes)


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
        mp_layer: str = "graphconv",
        output_dim=None,
    ):
        super().__init__()

        if model not in {
            "sag", "topk", "ndrp", "ndp", "graclus", "asapool", "pan", "cop",
            "cgi", "kmis", "gsap", "hgpsl", "hdpsl", "pars"
        }:
            raise ValueError(
                "model must be one of 'sag', 'topk', 'ndrp', 'ndp', "
                "'graclus', 'asapool', 'pan', 'cop', 'cgi', 'kmis', 'gsap', 'hgpsl', or 'pars'"
            )

        self.conv1 = make_message_passing_layer(mp_layer, input_dim, hidden)
        self.conv2 = make_message_passing_layer(mp_layer, hidden, hidden)
        self.conv3 = make_message_passing_layer(mp_layer, hidden, hidden)

        if model == "sag":
            pooling_layer = SAGPooling
        elif model == "topk":
            pooling_layer = TopKPooling
        elif model == "ndrp":
            pooling_layer = NDRPPooling
        elif model == "graclus":
            pooling_layer = GraclusPooling
        elif model == "asapool":
            pooling_layer = lambda channels, ratio: ASAPoolingAdapter(
                channels,
                ratio,
                mp_layer,
            )
        elif model == "pan":
            pooling_layer = PANPoolingAdapter
        elif model == "cop":
            pooling_layer = lambda channels, ratio: CoPoolingAdapter(
                channels, ratio, mp_layer
            )
        elif model == "cgi":
            pooling_layer = CGIPool
        elif model == "kmis":
            # Researcher-requested KMIS default: K=3.
            pooling_layer = lambda channels, ratio: KMISPool(channels, k=3)
        elif model == "gsap":
            pooling_layer = lambda channels, ratio: GSAPool(
                channels, pooling_ratio=ratio, alpha=0.4
            )
        elif model in {"hgpsl", "hdpsl"}:
            pooling_layer = lambda channels, ratio: HGPSLPool(
                channels, ratio=ratio
            )
        elif model == "pars":
            pooling_layer = lambda channels, ratio: ParsPooling(
                channels, ratio=ratio
            )
        else:
            pooling_layer = NDPPooling
        self.pool1 = pooling_layer(hidden, ratio=pratio)
        self.pool2 = pooling_layer(hidden, ratio=pratio)

        self.model = model
        self.mp_layer = mp_layer
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            2 * hidden, num_classes if output_dim is None else output_dim
        )
        self.last_pool_time = 0.0
        self._pool_events = []
        self._pool_cpu_time = 0.0

    def _timed_pool(self, pool, x, edge_index, batch):
        if x.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = pool(x, edge_index, batch=batch)
            end_event.record()
            self._pool_events.append((start_event, end_event))
            return result

        pool_start = time.perf_counter()
        result = pool(x, edge_index, batch=batch)
        self._pool_cpu_time += time.perf_counter() - pool_start
        return result

    def finish_pool_timing(self):
        """Resolve asynchronous CUDA pool timings for the current window."""
        if self._pool_events:
            torch.cuda.synchronize()
            self.last_pool_time = sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._pool_events
            )
            self._pool_events.clear()
        else:
            self.last_pool_time = self._pool_cpu_time
        self._pool_cpu_time = 0.0

    def reset_pool_timing(self):
        """Start a new pooling-timing measurement window."""
        self._pool_events.clear()
        self._pool_cpu_time = 0.0
        self.last_pool_time = 0.0

    def forward(self, x, edge_index, batch):
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


class CountSketchPooling(nn.Module):
    """Three-GCN classifier using sparse CountSketch pooling stages."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        q: int,
        hidden: int = 32,
        pratio: float = 0.5,
        dropout: float = 0.2,
        mp_layer: str = "graphconv",
        output_dim=None,
    ):
        super().__init__()
        if q not in {1, 2, 4}:
            raise ValueError("CountSketch q must be 1, 2, or 4")

        self.q = q
        self.pratio = pratio
        self.mp_layer = mp_layer
        self.conv1 = make_message_passing_layer(mp_layer, input_dim, hidden)
        self.conv2 = make_message_passing_layer(mp_layer, hidden, hidden)
        self.conv3 = make_message_passing_layer(mp_layer, hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            2 * hidden, num_classes if output_dim is None else output_dim
        )
        self.last_pool_time = 0.0
        self._pool_events = []
        self._pool_cpu_time = 0.0

    def _timed_pool(self, x, edge_index, edge_weight, batch):
        if x.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = countsketch_pool(
                x,
                edge_index,
                self.pratio,
                self.q,
                batch=batch,
                edge_weight=edge_weight,
            )
            end_event.record()
            self._pool_events.append((start_event, end_event))
            return result

        pool_start = time.perf_counter()
        result = countsketch_pool(
            x,
            edge_index,
            self.pratio,
            self.q,
            batch=batch,
            edge_weight=edge_weight,
        )
        self._pool_cpu_time += time.perf_counter() - pool_start
        return result

    def finish_pool_timing(self):
        """Resolve asynchronous CUDA pool timings for the current window."""
        if self._pool_events:
            torch.cuda.synchronize()
            self.last_pool_time = sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._pool_events
            )
            self._pool_events.clear()
        else:
            self.last_pool_time = self._pool_cpu_time
        self._pool_cpu_time = 0.0

    def reset_pool_timing(self):
        """Start a new pooling-timing measurement window."""
        self._pool_events.clear()
        self._pool_cpu_time = 0.0
        self.last_pool_time = 0.0

    def forward(self, x, edge_index, batch, edge_weight=None):
        # Selected message-passing layer: input_dim -> hidden.
        x = self.conv1(x, edge_index, edge_weight)
        # ReLU.
        x = x.relu()
        # Dropout.
        x = self.dropout(x)
        # Pooling: ratio = pratio.
        x, edge_index, edge_weight, batch = self._timed_pool(
            x, edge_index, edge_weight, batch
        )

        # Selected message-passing layer: hidden -> hidden.
        x = self.conv2(x, edge_index, edge_weight)
        # ReLU.
        x = x.relu()
        # Dropout.
        x = self.dropout(x)
        # Pooling: ratio = pratio.
        x, edge_index, edge_weight, batch = self._timed_pool(
            x, edge_index, edge_weight, batch
        )

        # Selected message-passing layer: hidden -> hidden.
        x = self.conv3(x, edge_index, edge_weight)
        # ReLU.
        x = x.relu()
        # Dropout.
        x = self.dropout(x)
        # Readout: element-wise mean.
        mean = global_mean_pool(x, batch)
        # Readout: element-wise max.
        maximum = global_max_pool(x, batch)
        # 2 * hidden-dimensional graph representation.
        graph_representation = torch.cat([mean, maximum], dim=1)
        # Linear output layer.
        return self.classifier(graph_representation)


EPS = 1e-15


def rank3_diag(x):
    eye = torch.eye(x.size(1), device=x.device, dtype=x.dtype)
    return eye * x.unsqueeze(2).expand(*x.size(), x.size(1))


def rank3_trace(x):
    return torch.einsum("ijj->i", x)


def dense_hosc_pool(x, adj, s, mask=None, mu=0.5, alpha=0.5):
    """HoscPool from the benchmark paper, used with dense batches."""
    s = torch.softmax(s, dim=-1)
    batch_size, num_nodes, _ = x.size()
    k = s.size(-1)
    if mask is not None:
        valid = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * valid, s * valid
    out = s.transpose(1, 2) @ x
    out_adj = s.transpose(1, 2) @ adj @ s
    motif_adj = (adj @ adj) * adj
    motif_out_adj = s.transpose(1, 2) @ motif_adj @ s
    mincut_loss = x.new_zeros(())
    hosc_loss = x.new_zeros(())
    if alpha < 1:
        sas = torch.einsum("ijj->ij", out_adj)
        degree = rank3_diag(torch.einsum("ijk->ij", adj))
        sds = s.transpose(1, 2) @ degree @ s
        sds = torch.einsum("ijj->ij", sds) + EPS
        mincut_loss = -(sas / sds).sum(dim=1).mean() / k
    if alpha > 0:
        sas = torch.einsum("ijj->ij", motif_out_adj)
        motif_degree = rank3_diag(torch.einsum("ijk->ij", motif_adj))
        sds = s.transpose(1, 2) @ motif_degree @ s
        sds = torch.einsum("ijj->ij", sds) + EPS
        hosc_loss = -(sas / sds).sum(dim=1).mean() / k
    hosc_loss = (1 - alpha) * mincut_loss + alpha * hosc_loss
    ss = s.transpose(1, 2) @ s
    identity = torch.eye(k, device=x.device, dtype=x.dtype)
    ortho = torch.linalg.norm(
        ss / torch.linalg.norm(ss, dim=(-1, -2), keepdim=True)
        - identity / torch.linalg.norm(identity),
        dim=(-1, -2),
    ).mean()
    out_adj = out_adj.clone()
    ind = torch.arange(k, device=x.device)
    out_adj[:, ind, ind] = 0
    degree = torch.sqrt(torch.einsum("ijk->ij", out_adj) + EPS)[:, :, None]
    out_adj = (out_adj / degree) / degree.transpose(1, 2)
    return out, out_adj, hosc_loss, mu * ortho


def dense_just_balance_pool(x, adj, s, mask=None):
    """Just Balance pooling from the benchmark paper."""
    s = torch.softmax(s, dim=-1)
    batch_size, num_nodes, _ = x.size()
    k = s.size(-1)
    if mask is not None:
        valid = mask.view(batch_size, num_nodes, 1).to(x.dtype)
        x, s = x * valid, s * valid
    out = s.transpose(1, 2) @ x
    out_adj = s.transpose(1, 2) @ adj @ s
    loss = -(rank3_trace(torch.sqrt(s.transpose(1, 2) @ s + EPS))).mean()
    loss = loss / torch.sqrt(x.new_tensor(float(num_nodes * k)))
    out_adj = out_adj.clone()
    ind = torch.arange(k, device=x.device)
    out_adj[:, ind, ind] = 0
    degree = torch.sqrt(torch.einsum("ijk->ij", out_adj) + EPS)[:, :, None]
    out_adj = (out_adj / degree) / degree.transpose(1, 2)
    return out, out_adj, loss


class DensePoolingStage(nn.Module):
    """One dense pooling stage for paper and baseline methods."""

    def __init__(self, hidden: int, num_clusters: int, model: str, pratio: float):
        super().__init__()
        self.model = model
        self.pratio = pratio
        self.num_clusters = num_clusters
        self.assignment = DenseGCNConv(hidden, num_clusters)
        self.dmon = DMoNPooling([hidden, hidden], num_clusters) if model == "dmon" else None

    def forward(self, x, adj, mask):
        if self.model in {"gaus", "unif", "count1", "count2", "count4"}:
            x, adj = random_dense_pool(
                x,
                adj,
                self.pratio,
                self.model,
            )
            pooled_mask = x.new_ones(x.size(0), x.size(1), dtype=torch.bool)
            return x, adj, pooled_mask, x.new_zeros(())

        if self.model == "dmon":
            _, pooled_x, pooled_adj, spectral_loss, ortho_loss, cluster_loss = self.dmon(
                x, adj, mask=mask
            )
            pooled_mask = x.new_ones(
                pooled_x.size(0), pooled_x.size(1), dtype=torch.bool
            )
            return pooled_x, pooled_adj, pooled_mask, spectral_loss + ortho_loss + cluster_loss

        assignment = self.assignment(x, adj, mask=mask)

        if self.model == "hosc":
            pooled_x, pooled_adj, hosc_loss, ortho_loss = dense_hosc_pool(
                x, adj, assignment, mask=mask, mu=0.5, alpha=0.5
            )
            pooled_mask = x.new_ones(
                pooled_x.size(0), pooled_x.size(1), dtype=torch.bool
            )
            return pooled_x, pooled_adj, pooled_mask, hosc_loss + ortho_loss

        if self.model == "justb":
            pooled_x, pooled_adj, balance_loss = dense_just_balance_pool(
                x, adj, assignment, mask=mask
            )
            pooled_mask = x.new_ones(
                pooled_x.size(0), pooled_x.size(1), dtype=torch.bool
            )
            return pooled_x, pooled_adj, pooled_mask, balance_loss

        if self.model == "diff":
            x, adj, link_loss, entropy_loss = dense_diff_pool(
                x, adj, assignment, mask=mask
            )
            auxiliary_loss = link_loss + entropy_loss
        else:
            x, adj, mincut_loss, orthogonality_loss = dense_mincut_pool(
                x, adj, assignment, mask=mask
            )
            auxiliary_loss = mincut_loss + orthogonality_loss

        pooled_mask = x.new_ones(x.size(0), x.size(1), dtype=torch.bool)
        return x, adj, pooled_mask, auxiliary_loss


def random_dense_pool(x, adj, pratio, model):
    """Apply a batched Gaussian, Uniform, or CountSketch pooling operator."""
    batch_size, num_nodes, _ = x.size()
    num_clusters = max(1, int(round(pratio * num_nodes)))

    if model == "gaus":
        # St = S.T with entries sampled from N(0, 1 / p).
        st = x.new_empty(batch_size, num_nodes, num_clusters).normal_(
            0.0, num_clusters ** -0.5
        )
    elif model == "unif":
        # St = S.T with entries sampled uniformly from [-sqrt(3/p), sqrt(3/p)].
        bound = (3.0 / num_clusters) ** 0.5
        st = x.new_empty(batch_size, num_nodes, num_clusters).uniform_(
            -bound, bound
        )
    else:
        q = {"count1": 1, "count2": 2, "count4": 4}[model]
        return countsketch_dense_pool(x, adj, num_clusters, q)

    s = st.transpose(1, 2)
    ast = torch.bmm(adj, st)
    return torch.bmm(s, x), torch.bmm(s, ast)


def countsketch_dense_pool(x, adj, num_clusters, q):
    """Batched version of the original CountSketch pooling function.

    The original function constructs a sketch matrix ``S`` and returns::

        X' = S @ X
        A' = S @ A @ S.T

    This implementation performs the same two mathematical operations, but
    stores only the non-zero cluster assignments and signs.  The change is
    needed because the dense ``S`` matrix contains mostly zeros for Count1,
    Count2, and Count4.  The input is also batched, so tensors have a leading
    batch dimension that is absent from the original single-graph function.
    """
    batch_size, num_nodes, feature_dim = x.shape
    if q > num_clusters:
        raise ValueError(
            f"CountSketch q={q} cannot exceed {num_clusters} clusters"
        )

    # Original code: choose q cluster locations for every node before building S.T.
    # New code: store those locations directly instead of constructing dense S.T.
    rows = torch.randint(
        num_clusters,
        (batch_size, num_nodes, q),
        device=x.device,
    )

    # Original q>1 code uses topk to sample q distinct locations without replacement.
    # Resampling only duplicates preserves that same distinct-assignment requirement.
    for assignment in range(1, q):
        duplicate = (
            rows[:, :, assignment, None] == rows[:, :, :assignment]
        ).any(dim=-1)
        while duplicate.any():
            replacement = torch.randint(
                num_clusters,
                duplicate.shape,
                device=x.device,
            )
            rows[:, :, assignment] = torch.where(
                duplicate,
                replacement,
                rows[:, :, assignment],
            )
            duplicate = (
                rows[:, :, assignment, None] == rows[:, :, :assignment]
            ).any(dim=-1)

    # Original code: assign independent Rademacher signs to the non-zero entries of S.
    # The signs are stored directly because S itself is no longer materialized.
    scale = q ** -0.5
    signs = x.new_empty(batch_size, num_nodes, q).random_(2)
    signs = signs.mul_(2 * scale).sub_(scale)

    # The original function handles one graph, while this model handles B graphs.
    # Offsets keep cluster IDs from different graphs independent during batching.
    graph_offsets = (
        torch.arange(batch_size, device=x.device).view(batch_size, 1, 1)
        * num_clusters
    )
    flat_clusters = (rows + graph_offsets).reshape(-1)

    # Original step: X' = S @ X.
    # Equivalent batched step: add each signed node feature to its assigned cluster.
    pooled_x = x.new_zeros(batch_size * num_clusters, feature_dim)
    signed_x = (x.unsqueeze(2) * signs.unsqueeze(-1)).reshape(-1, feature_dim)
    pooled_x.index_add_(0, flat_clusters, signed_x)
    pooled_x = pooled_x.view(batch_size, num_clusters, feature_dim)

    # Original first adjacency step: compute S @ A.
    # Each signed adjacency row is accumulated into its assigned cluster.
    pooled_left = x.new_zeros(batch_size * num_clusters, num_nodes)
    for assignment in range(q):
        cluster_ids = (
            rows[:, :, assignment] + graph_offsets[:, :, 0]
        ).reshape(-1)
        signed_rows = adj * signs[:, :, assignment].unsqueeze(-1)
        pooled_left.index_add_(0, cluster_ids, signed_rows.reshape(-1, num_nodes))
    pooled_left = pooled_left.view(batch_size, num_clusters, num_nodes)

    # Original second adjacency step: compute (S @ A) @ S.T.
    # Scatter-add performs the same column assignment without constructing S.T.
    pooled_adj = x.new_zeros(batch_size, num_clusters, num_clusters)
    for assignment in range(q):
        values = pooled_left * signs[:, :, assignment].unsqueeze(1)
        target_clusters = rows[:, :, assignment].unsqueeze(1).expand(
            batch_size, num_clusters, num_nodes
        )
        pooled_adj.scatter_add_(2, target_clusters, values)

    return pooled_x, pooled_adj


class DensePool(nn.Module):
    """Dense counterpart with the same architecture as SparsePooling."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        model: str = "diff",
        hidden: int = 32,
        pratio: float = 0.5,
        dropout: float = 0.2,
        max_nodes: int = 500,
        output_dim=None,
    ):
        super().__init__()

        dense_models = {
            "diff", "mincut", "gaus", "unif",
            "count1", "count2", "count4",
            "dmon", "hosc", "justb",
        }
        if model not in dense_models:
            raise ValueError(
                "model must be one of diff, mincut, gaus, unif, "
                "count1, count2, count4, dmon, hosc, or justb"
            )

        if model == "dmon":
            # Researcher-requested DMoN default: four clusters at each stage.
            num_clusters = 4
            second_num_clusters = 4
        else:
            num_clusters = max(1, int(round(pratio * max_nodes)))
            second_num_clusters = max(1, int(round(pratio * num_clusters)))
        self.max_nodes = max_nodes
        self.conv1 = DenseGCNConv(input_dim, hidden)
        self.conv2 = DenseGCNConv(hidden, hidden)
        self.conv3 = DenseGCNConv(hidden, hidden)
        self.pool1 = DensePoolingStage(hidden, num_clusters, model, pratio)
        self.pool2 = DensePoolingStage(
            hidden, second_num_clusters, model, pratio
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(
            2 * hidden, num_classes if output_dim is None else output_dim
        )
        self.last_pool_time = 0.0
        self.last_auxiliary_loss = None
        self._pool_events = []
        self._pool_cpu_time = 0.0

    def _timed_pool(self, pool, x, adj, mask):
        if x.is_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            result = pool(x, adj, mask)
            end_event.record()
            self._pool_events.append((start_event, end_event))
            return result

        pool_start = time.perf_counter()
        result = pool(x, adj, mask)
        self._pool_cpu_time += time.perf_counter() - pool_start
        return result

    def finish_pool_timing(self):
        """Resolve asynchronous CUDA pool timings for the current window."""
        if self._pool_events:
            torch.cuda.synchronize()
            self.last_pool_time = sum(
                start.elapsed_time(end) / 1000.0
                for start, end in self._pool_events
            )
            self._pool_events.clear()
        else:
            self.last_pool_time = self._pool_cpu_time
        self._pool_cpu_time = 0.0

    def reset_pool_timing(self):
        """Start a new pooling-timing measurement window."""
        self._pool_events.clear()
        self._pool_cpu_time = 0.0
        self.last_pool_time = 0.0

    def forward(self, x, edge_index=None, batch=None, adj=None, mask=None):
        # Input preparation: use precomputed dense tensors when available.
        if adj is None or mask is None:
            x, mask = to_dense_batch(x, batch, max_num_nodes=self.max_nodes)
            adj = to_dense_adj(
                edge_index,
                batch=batch,
                max_num_nodes=self.max_nodes,
            )

        # GCNConv: input_dim -> hidden.
        x = self.conv1(x, adj, mask=mask)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = pratio.
        x, adj, mask, auxiliary_loss_1 = self._timed_pool(
            self.pool1, x, adj, mask
        )

        # GCNConv: hidden -> hidden.
        x = self.conv2(x, adj, mask=mask)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Pooling: ratio = pratio.
        x, adj, mask, auxiliary_loss_2 = self._timed_pool(
            self.pool2, x, adj, mask
        )

        # GCNConv: hidden -> hidden.
        x = self.conv3(x, adj, mask=mask)

        # ReLU
        x = x.relu()

        # Dropout
        x = self.dropout(x)

        # Readout: element-wise mean.
        valid = mask.unsqueeze(-1)
        mean = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)

        # Readout: element-wise max.
        maximum = x.masked_fill(~valid, float("-inf")).max(dim=1).values

        # 2 * hidden-dimensional graph representation.
        graph_representation = torch.cat([mean, maximum], dim=1)

        # Linear output layer.
        output = self.classifier(graph_representation)
        self.last_auxiliary_loss = auxiliary_loss_1 + auxiliary_loss_2
        return output
