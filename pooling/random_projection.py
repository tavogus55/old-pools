import torch


def uniform_pool(X: torch.Tensor, A: torch.Tensor, pr: float):
    """Uniform node clustering graph pooling."""
    n = X.size(0)
    p = max(1, int(round(pr * n)))

    # Sample St = S.T directly to avoid explicitly materializing S first.
    bound = (3.0 / p) ** 0.5
    St = X.new_empty(n, p).uniform_(-bound, bound)
    S = St.t()

    ASt = torch.sparse.mm(A, St) if A.is_sparse else torch.mm(A, St)
    return torch.mm(S, X), torch.mm(S, ASt)


def batch_uniform_pool(X, A, mask, pr):
    """Mask-aware batched uniform pooling with no per-graph Python loop.

    Each active assignment entry for graph g is sampled independently from
    Uniform[-sqrt(3/p_g), sqrt(3/p_g)]. Padded input nodes and unused pooled
    rows are masked out of X' = SX and A' = SAS^T.
    """
    batch_size, max_n, _ = X.shape
    nodes_per_graph = mask.sum(dim=1)
    pooled_per_graph = torch.round(nodes_per_graph * pr).long().clamp(min=1)
    max_p = int(pooled_per_graph.max())

    pooled_mask = (
        torch.arange(max_p, device=X.device).unsqueeze(0)
        < pooled_per_graph.unsqueeze(1)
    )
    assignment_mask = mask.unsqueeze(-1) & pooled_mask.unsqueeze(1)

    # Draw U[-1, 1], then apply each graph's sqrt(3/p_g) bound.
    bound = (3.0 / pooled_per_graph.to(X.dtype)).sqrt().view(
        batch_size, 1, 1
    )
    St = X.new_empty((batch_size, max_n, max_p)).uniform_(-1.0, 1.0)
    St = St * bound * assignment_mask.to(X.dtype)
    S = St.transpose(1, 2)

    pooled_x = torch.bmm(S, X)
    ASt = torch.bmm(A, St)
    pooled_A = torch.bmm(S, ASt)

    pooled_x = pooled_x * pooled_mask.unsqueeze(-1).to(X.dtype)
    pooled_A = pooled_A * (
        pooled_mask.unsqueeze(1) & pooled_mask.unsqueeze(2)
    ).to(A.dtype)
    return pooled_x, pooled_A, pooled_mask


def gaussian_pool(X: torch.Tensor, A: torch.Tensor, pr: float):
    """Gaussian node clustering graph pooling."""
    n = X.size(0)
    p = max(1, int(round(pr * n)))

    # Entries of S.T are i.i.d. N(0, 1/p).
    St = X.new_empty(n, p).normal_(0.0, p ** -0.5)
    S = St.t()

    ASt = torch.sparse.mm(A, St) if A.is_sparse else torch.mm(A, St)
    return torch.mm(S, X), torch.mm(S, ASt)


def batch_gaussian_pool(X, A, mask, pr):
    """Mask-aware batched Gaussian pooling with no per-graph Python loop.

    Each graph still receives an independent matrix whose active entries are
    i.i.d. N(0, 1/p_g), and computes X' = SX and A' = SAS^T. Rows and columns
    belonging to batch padding are masked out of the contractions.
    """
    batch_size, max_n, _ = X.shape
    nodes_per_graph = mask.sum(dim=1)
    pooled_per_graph = torch.round(nodes_per_graph * pr).long().clamp(min=1)
    max_p = int(pooled_per_graph.max())

    node_mask = mask.unsqueeze(-1)
    pooled_mask = (
        torch.arange(max_p, device=X.device).unsqueeze(0)
        < pooled_per_graph.unsqueeze(1)
    )
    assignment_mask = node_mask & pooled_mask.unsqueeze(1)

    # St[b, i, j] ~ N(0, 1/p_b) for active node/cluster pairs.
    std = pooled_per_graph.to(X.dtype).rsqrt().view(batch_size, 1, 1)
    St = X.new_empty((batch_size, max_n, max_p)).normal_()
    St = St * std * assignment_mask.to(X.dtype)
    S = St.transpose(1, 2)

    # Batched equivalents of S @ X and (S @ A) @ S.T.
    pooled_x = torch.bmm(S, X)
    ASt = torch.bmm(A, St)
    pooled_A = torch.bmm(S, ASt)

    pooled_x = pooled_x * pooled_mask.unsqueeze(-1).to(X.dtype)
    pooled_A = pooled_A * (
        pooled_mask.unsqueeze(1) & pooled_mask.unsqueeze(2)
    ).to(A.dtype)
    return pooled_x, pooled_A, pooled_mask


def batch_dense_pool(X, A, mask, pr, method):
    """Apply a dense random projection pool to an entire padded batch."""
    if method == 'unif':
        return batch_uniform_pool(X, A, mask, pr)
    if method == 'gaus':
        return batch_gaussian_pool(X, A, mask, pr)
    raise ValueError(f'Unknown dense random pool: {method}')
