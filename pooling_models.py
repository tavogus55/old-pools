import time

import torch
from torch import nn
from torch_geometric.nn import (
    DenseGCNConv,
    GCNConv,
    SAGPooling,
    TopKPooling,
    global_max_pool,
    global_mean_pool,
    dense_diff_pool,
    dense_mincut_pool,
)
from torch_geometric.utils import subgraph, to_dense_adj, to_dense_batch
from torch_scatter import scatter_add, scatter_max, scatter_min


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

        if model not in {"sag", "topk", "ndrp", "ndp"}:
            raise ValueError("model must be one of 'sag', 'topk', 'ndrp', or 'ndp'")

        self.conv1 = GCNConv(input_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)

        if model == "sag":
            pooling_layer = SAGPooling
        elif model == "topk":
            pooling_layer = TopKPooling
        elif model == "ndrp":
            pooling_layer = NDRPPooling
        else:
            pooling_layer = NDPPooling
        self.pool1 = pooling_layer(hidden, ratio=pratio)
        self.pool2 = pooling_layer(hidden, ratio=pratio)

        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden, num_classes)
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


class DensePoolingStage(nn.Module):
    """One dense DiffPool or MinCutPool stage."""

    def __init__(self, hidden: int, num_clusters: int, model: str, pratio: float):
        super().__init__()
        self.model = model
        self.pratio = pratio
        self.num_clusters = num_clusters
        self.assignment = DenseGCNConv(hidden, num_clusters)

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

        assignment = self.assignment(x, adj, mask=mask)

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
    ):
        super().__init__()

        dense_models = {
            "diff", "mincut", "gaus", "unif",
            "count1", "count2", "count4",
        }
        if model not in dense_models:
            raise ValueError(
                "model must be one of diff, mincut, gaus, unif, "
                "count1, count2, or count4"
            )

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
        self.classifier = nn.Linear(2 * hidden, num_classes)
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
