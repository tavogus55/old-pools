import torch
import torch_geometric.data
import torch_geometric.transforms as T

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader


def _register_safe_globals():
    safe = [torch_geometric.data.data.Data]

    for name in ['DataEdgeAttr', 'DataTensorAttr']:
        if hasattr(torch_geometric.data.data, name):
            safe.append(getattr(torch_geometric.data.data, name))

    if hasattr(torch_geometric.data, 'storage'):
        for name in ['GlobalStorage', 'NodeStorage', 'EdgeStorage']:
            if hasattr(torch_geometric.data.storage, name):
                safe.append(getattr(torch_geometric.data.storage, name))

    # torch.serialization.add_safe_globals(safe)


_register_safe_globals()


def load_tu_graphs(root='data', name='PROTEINS', max_nodes=None, quantile=0.95, use_node_attr=True):
    dataset = TUDataset(root=root, name=name, use_node_attr=use_node_attr)

    add_const = T.Constant(value=1.0, cat=False) if dataset.num_features == 0 else None

    sizes = torch.tensor([data.num_nodes for data in dataset], dtype=torch.float)
    if max_nodes is None:
        max_nodes = max(1, int(torch.quantile(sizes, quantile).item()))

    to_dense = T.ToDense(max_nodes)

    sparse_graphs = []
    dense_graphs = []

    for data in dataset:
        data = data.clone()

        if add_const is not None:
            data = add_const(data)

        if data.num_nodes > max_nodes:
            continue

        sparse_graphs.append(data)

        dense_data = data.clone()

        # 关键修复：去掉 edge_attr，避免 ToDense 生成 [N, N, F_e]
        if getattr(dense_data, 'edge_attr', None) is not None:
            dense_data.edge_attr = None

        dense = to_dense(dense_data)

        # 保险处理：若仍有额外边特征维，则压成二值邻接
        if dense.adj.dim() == 3:
            dense.adj = (dense.adj.abs().sum(dim=-1) > 0).float()

        dense_graphs.append(dense)

    if len(sparse_graphs) == 0:
        raise ValueError(f'No graph left after filtering with max_nodes={max_nodes}.')

    in_channels = sparse_graphs[0].num_features
    num_classes = dataset.num_classes

    return sparse_graphs, dense_graphs, in_channels, num_classes, max_nodes


def _select(graphs, indices):
    return [graphs[i] for i in indices]


def make_loaders(
    sparse_graphs,
    dense_graphs,
    split_indices,
    batch_size=32,
):
    train_idx, val_idx, test_idx = split_indices

    sparse_train = _select(sparse_graphs, train_idx)
    sparse_val = _select(sparse_graphs, val_idx)
    sparse_test = _select(sparse_graphs, test_idx)

    dense_train = _select(dense_graphs, train_idx)
    dense_val = _select(dense_graphs, val_idx)
    dense_test = _select(dense_graphs, test_idx)

    sparse_loaders = (
        DataLoader(sparse_train, batch_size=batch_size, shuffle=True),
        DataLoader(sparse_val, batch_size=batch_size, shuffle=False),
        DataLoader(sparse_test, batch_size=batch_size, shuffle=False),
    )

    dense_loaders = (
        DenseDataLoader(dense_train, batch_size=batch_size, shuffle=True),
        DenseDataLoader(dense_val, batch_size=batch_size, shuffle=False),
        DenseDataLoader(dense_test, batch_size=batch_size, shuffle=False),
    )

    return sparse_loaders, dense_loaders
