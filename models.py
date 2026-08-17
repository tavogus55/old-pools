import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    DenseGCNConv,
    GCNConv,
    SAGPooling,
    TopKPooling,
    dense_diff_pool,
    dense_mincut_pool,
    global_max_pool,
    global_mean_pool,
)

from pooling import (
    batch_dense_pool,
    sparse_countsketch_pool,
    sparse_graclus_pool,
    sparse_ndp_pool,
    sparse_node_drop_pool,
)


HIDDEN_CHANNELS = 32

AVAILABLE_METHODS = (
    'mean',
    'unif',
    'gaus',
    'ndrp',
    'ndp',
    'graclus',
    'count1',
    'count2',
    'count4',
    'topk',
    'sag',
    'diff',
    'mincut',
)

DENSE_RANDOM_METHODS = {'unif', 'gaus'}
SPARSE_COUNTSKETCH_METHODS = {'count1', 'count2', 'count4'}
DENSE_METHODS = {'diff', 'mincut', *DENSE_RANDOM_METHODS}


class SparsePoolNet(nn.Module):
    """Shared sparse backbone: GCN -> pool -> GCN -> pool -> GCN -> readout."""

    def __init__(self, method, in_channels, num_classes, pool_ratio=0.5, dropout=0.2):
        super().__init__()
        self.method = method
        self.pool_ratio = pool_ratio
        self.conv1 = GCNConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = GCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * HIDDEN_CHANNELS, num_classes)

        if method == 'topk':
            self.pool1 = TopKPooling(HIDDEN_CHANNELS, ratio=pool_ratio)
            self.pool2 = TopKPooling(HIDDEN_CHANNELS, ratio=pool_ratio)
        elif method == 'sag':
            self.pool1 = SAGPooling(HIDDEN_CHANNELS, ratio=pool_ratio)
            self.pool2 = SAGPooling(HIDDEN_CHANNELS, ratio=pool_ratio)

    def _pool(self, stage, x, edge_index, batch):
        if self.method == 'mean':
            # The mean baseline has no graph-coarsening operator. Its two pool
            # positions are identity operations; mean is applied at readout.
            return x, edge_index, batch

        if self.method in {'topk', 'sag'}:
            pool = self.pool1 if stage == 1 else self.pool2
            x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
            return x, edge_index, batch

        if self.method == 'ndrp':
            return sparse_node_drop_pool(
                x, edge_index, batch, self.pool_ratio
            )

        if self.method == 'ndp':
            return sparse_ndp_pool(x, edge_index, batch, self.pool_ratio)

        if self.method == 'graclus':
            return sparse_graclus_pool(x, edge_index, batch)

        raise ValueError(f'Unsupported sparse pooling method: {self.method}')

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch = self._pool(1, x, edge_index, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, batch = self._pool(2, x, edge_index, batch)

        x = F.relu(self.conv3(x, edge_index))
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)
        x = self.lin(self.dropout(x))

        aux_loss = x.new_zeros(())
        return F.log_softmax(x, dim=-1), aux_loss


class DenseSignedGCNConv(nn.Module):
    """Dense GCN convolution whose degree uses absolute signed edge mass."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, adj, mask=None):
        size = adj.size(-1)
        identity = torch.eye(size, dtype=adj.dtype, device=adj.device).expand_as(adj)
        adj = adj + identity
        degree = adj.abs().sum(dim=-1).clamp_min(1e-12)
        degree_inv_sqrt = degree.rsqrt()
        norm_adj = degree_inv_sqrt.unsqueeze(-1) * adj * degree_inv_sqrt.unsqueeze(-2)
        out = norm_adj @ self.lin(x)
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class SparseSignedGCNConv(nn.Module):
    """Sparse counterpart of the absolute-degree signed GCN normalization."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(
            in_channels,
            out_channels,
            normalize=False,
            add_self_loops=False,
        )

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        loops = torch.arange(num_nodes, device=x.device)
        loop_index = torch.stack([loops, loops])
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_weight = torch.cat([edge_weight, x.new_ones(num_nodes)])

        _, target = edge_index
        degree = x.new_zeros(num_nodes)
        degree.index_add_(0, target, edge_weight.abs())
        degree_inv_sqrt = degree.clamp_min(1e-12).rsqrt()
        source, target = edge_index
        normalized_weight = (
            degree_inv_sqrt[source] * edge_weight * degree_inv_sqrt[target]
        )
        return self.conv(x, edge_index, normalized_weight)


class SparseCountSketchNet(nn.Module):
    """Shared backbone with native sparse CountSketch at both pool positions."""

    def __init__(self, method, in_channels, num_classes, pool_ratio=0.5, dropout=0.2):
        super().__init__()
        self.q = int(method.removeprefix('count'))
        self.pool_ratio = pool_ratio
        self.conv1 = GCNConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = SparseSignedGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = SparseSignedGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * HIDDEN_CHANNELS, num_classes)

    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        batch = data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, batch, edge_weight = sparse_countsketch_pool(
            x, edge_index, batch, self.pool_ratio, self.q
        )

        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x, edge_index, batch, edge_weight = sparse_countsketch_pool(
            x, edge_index, batch, self.pool_ratio, self.q, edge_weight
        )

        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=-1)
        x = self.lin(self.dropout(x))

        aux_loss = x.new_zeros(())
        return F.log_softmax(x, dim=-1), aux_loss


def dense_mean_max_readout(x, mask):
    mask_expanded = mask.unsqueeze(-1)
    mean = (x * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
    maximum = x.masked_fill(~mask_expanded, float('-inf')).max(dim=1).values
    return torch.cat([mean, maximum], dim=-1)


class DenseRandomPoolNet(nn.Module):
    """Fixed backbone adapted to dense random projection and node-drop pools."""

    def __init__(self, method, in_channels, num_classes, pool_ratio=0.5, dropout=0.2):
        super().__init__()
        self.method = method
        self.pool_ratio = pool_ratio
        self.conv1 = DenseGCNConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = DenseSignedGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = DenseSignedGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * HIDDEN_CHANNELS, num_classes)

    def forward(self, data):
        x = data.x.float()
        adj = data.adj.float()
        mask = data.mask

        # DenseDataLoader pads to the dataset-wide maximum. Trim each batch to
        # its largest real graph so dense convolutions and projections do not
        # process columns that are padding for every graph in this batch.
        batch_max_nodes = int(mask.sum(dim=1).max())
        x = x[:, :batch_max_nodes]
        adj = adj[:, :batch_max_nodes, :batch_max_nodes]
        mask = mask[:, :batch_max_nodes]

        x = F.relu(self.conv1(x, adj, mask))
        x, adj, mask = batch_dense_pool(x, adj, mask, self.pool_ratio, self.method)

        x = F.relu(self.conv2(x, adj, mask))
        x, adj, mask = batch_dense_pool(x, adj, mask, self.pool_ratio, self.method)

        x = F.relu(self.conv3(x, adj, mask))
        x = dense_mean_max_readout(x, mask)
        x = self.lin(self.dropout(x))

        aux_loss = x.new_zeros(())
        return F.log_softmax(x, dim=-1), aux_loss


class DiffPoolNet(nn.Module):
    """Two-stage DiffPool with the same 32-wide three-convolution backbone."""

    def __init__(self, in_channels, num_classes, max_nodes, pool_ratio=0.5, dropout=0.2):
        super().__init__()
        clusters1 = max(1, math.ceil(max_nodes * pool_ratio))
        clusters2 = max(1, math.ceil(clusters1 * pool_ratio))

        self.conv1 = DenseGCNConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = DenseGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = DenseGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)

        # These convolutions are part of the selected DiffPool operators and
        # learn their assignment matrices; they are not backbone convolutions.
        self.assign1 = DenseGCNConv(HIDDEN_CHANNELS, clusters1)
        self.assign2 = DenseGCNConv(HIDDEN_CHANNELS, clusters2)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * HIDDEN_CHANNELS, num_classes)

    def forward(self, data):
        x = data.x.float()
        adj = data.adj.float()
        mask = data.mask

        x = F.relu(self.conv1(x, adj, mask))
        assignment = self.assign1(x, adj, mask)
        x, adj, link1, ent1 = dense_diff_pool(x, adj, assignment, mask)

        x = F.relu(self.conv2(x, adj))
        assignment = self.assign2(x, adj)
        x, adj, link2, ent2 = dense_diff_pool(x, adj, assignment)

        x = F.relu(self.conv3(x, adj))
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        x = self.lin(self.dropout(x))

        aux_loss = link1 + ent1 + link2 + ent2
        return F.log_softmax(x, dim=-1), aux_loss


class MinCutPoolNet(nn.Module):
    """Two-stage PyG MinCutPool with the shared 32-wide backbone."""

    def __init__(self, in_channels, num_classes, max_nodes, pool_ratio=0.5, dropout=0.2):
        super().__init__()
        clusters1 = max(1, math.ceil(max_nodes * pool_ratio))
        clusters2 = max(1, math.ceil(clusters1 * pool_ratio))

        self.conv1 = DenseGCNConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = DenseGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = DenseGCNConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        # Assignment GCNs belong to the two selected MinCutPool operators.
        self.assign1 = DenseGCNConv(HIDDEN_CHANNELS, clusters1)
        self.assign2 = DenseGCNConv(HIDDEN_CHANNELS, clusters2)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(2 * HIDDEN_CHANNELS, num_classes)

    def forward(self, data):
        x = data.x.float()
        adj = data.adj.float()
        mask = data.mask

        x = F.relu(self.conv1(x, adj, mask))
        assignment = self.assign1(x, adj, mask)
        x, adj, mincut1, ortho1 = dense_mincut_pool(x, adj, assignment, mask)

        x = F.relu(self.conv2(x, adj))
        assignment = self.assign2(x, adj)
        x, adj, mincut2, ortho2 = dense_mincut_pool(x, adj, assignment)

        x = F.relu(self.conv3(x, adj))
        x = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        x = self.lin(self.dropout(x))

        aux_loss = mincut1 + ortho1 + mincut2 + ortho2
        return F.log_softmax(x, dim=-1), aux_loss


def build_model(method, in_channels, hidden_channels, num_classes, max_nodes, pool_ratio,
                dropout=0.2):
    if hidden_channels != HIDDEN_CHANNELS:
        raise ValueError(
            f'All models use a fixed hidden width of {HIDDEN_CHANNELS}; '
            f'got hidden_channels={hidden_channels}.'
        )

    if method == 'diff':
        return DiffPoolNet(in_channels, num_classes, max_nodes, pool_ratio, dropout)

    if method == 'mincut':
        return MinCutPoolNet(in_channels, num_classes, max_nodes, pool_ratio, dropout)

    if method in DENSE_RANDOM_METHODS:
        return DenseRandomPoolNet(method, in_channels, num_classes, pool_ratio, dropout)

    if method in SPARSE_COUNTSKETCH_METHODS:
        return SparseCountSketchNet(method, in_channels, num_classes, pool_ratio, dropout)

    if method in AVAILABLE_METHODS:
        return SparsePoolNet(method, in_channels, num_classes, pool_ratio, dropout)

    raise ValueError(f'Unknown method: {method}')
