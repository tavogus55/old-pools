import argparse
import json
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.data
from sklearn.model_selection import KFold
# torch.serialization.add_safe_globals([torch_geometric.data.data.Data])

from data import load_tu_graphs, make_loaders
from models import AVAILABLE_METHODS, DENSE_METHODS, build_model
from utils import evaluate, get_logger, log_experiment_settings, save_to_csv, set_seed


MAX_NODES_BY_DATASET = {
    # TU datasets (benchmark settings shared with new-pools).
    'MUTAG': 150,
    'DD': 500,
    'IMDB-MULTI': 500,
    'PROTEINS': 700,
    'IMDB-BINARY': 500,
    'COLLAB': 150,
    'NCI1': 150,
    'NCI109': 150,
}


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def parse_methods(text):
    text = text.strip().lower()
    if text == 'all':
        return list(AVAILABLE_METHODS)

    methods = [x.strip() for x in text.split(',') if x.strip()]
    unknown = [m for m in methods if m not in AVAILABLE_METHODS]
    if unknown:
        raise ValueError(f'Unknown methods: {unknown}. Available: {AVAILABLE_METHODS}')
    return methods


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    start = time.perf_counter()

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(batch)
        y = batch.y.view(-1)

        loss = F.nll_loss(logits, y) + aux_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()

    train_time = time.perf_counter() - start
    avg_loss = total_loss / len(loader.dataset)

    return avg_loss, train_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PROTEINS')
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--methods', type=str, default='diff')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=[42, 43, 44, 45, 46, 47, 48, 49, 50, 51])
    parser.add_argument('--k-folds', '--k_folds', dest='k_folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden', type=int, choices=[32], default=32,
                        help='fixed backbone width (must be 32)')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--pool-ratio', type=float, default=0.5)
    parser.add_argument('--early-stop', type=int, default=2000)
    parser.add_argument('--tolerance', type=float, default=1e-4)
    parser.add_argument('--log-level', type=str, default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    parser.add_argument('--log-path', type=str, default='local')
    parser.add_argument('--exp-name', type=str, default='exp')
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_settings = {
        "logger": {
            "model": args.methods,
            "log_path": args.log_path,
            "dataset": args.dataset,
            "log_level": args.log_level.upper()
        },
    }
    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger(args.exp_name, timestamp)
    log_experiment_settings(logger, args)

    methods = parse_methods(args.methods)
    device = get_device()

    try:
        max_nodes = MAX_NODES_BY_DATASET[args.dataset]
    except KeyError as exc:
        supported = ', '.join(sorted(MAX_NODES_BY_DATASET))
        raise ValueError(
            f'Unsupported dataset {args.dataset!r}. '
            f'Fixed max-node settings are available for: {supported}'
        ) from exc

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    sparse_graphs, dense_graphs, in_channels, num_classes, max_nodes = load_tu_graphs(
        root=args.root,
        name=args.dataset,
        max_nodes=max_nodes,
        use_node_attr=True,
    )

    logger.info(
        f'Dataset loaded | name={args.dataset} | graphs={len(sparse_graphs)} | '
        f'in_channels={in_channels} | num_classes={num_classes} | '
        f'max_nodes={max_nodes} | device={device}'
    )

    if args.k_folds < 2:
        raise ValueError(f'k_folds must be at least 2, got {args.k_folds}')
    if args.k_folds > len(sparse_graphs):
        raise ValueError(
            f'k_folds ({args.k_folds}) cannot exceed the number of graphs '
            f'({len(sparse_graphs)})'
        )

    # Match duan-pooling: KFold selects the held-out test fold, then 10% of
    # the remaining folds is shuffled off for validation.
    kf = KFold(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=args.seeds[0],
    )
    random.seed(args.seeds[0])
    fold_splits = []
    indices = list(range(len(sparse_graphs)))
    for train_val_idx, test_idx in kf.split(indices):
        train_val_idx = list(train_val_idx)
        random.shuffle(train_val_idx)
        num_val = int(0.1 * len(train_val_idx))
        val_idx = train_val_idx[:num_val]
        train_idx = train_val_idx[num_val:]
        fold_splits.append((train_idx, val_idx, list(test_idx)))

    for method in methods:
        logger.info(f'Training method: {method}')

        best_val_accs = []
        best_test_accs = []
        best_test_macro_f1s = []
        times = []
        memories = []

        for fold_idx, split_indices in enumerate(fold_splits):
            sparse_loaders, dense_loaders = make_loaders(
                sparse_graphs=sparse_graphs,
                dense_graphs=dense_graphs,
                split_indices=split_indices,
                batch_size=args.batch_size,
            )

            if method in DENSE_METHODS:
                train_loader, val_loader, test_loader = dense_loaders
            else:
                train_loader, val_loader, test_loader = sparse_loaders

            logger.debug(
                f'Fold {fold_idx}: train={len(train_loader.dataset)}, '
                f'val={len(val_loader.dataset)}, test={len(test_loader.dataset)}'
            )

            for seed in args.seeds:
                set_seed(seed)

                model = build_model(
                    method=method,
                    in_channels=in_channels,
                    hidden_channels=args.hidden,
                    num_classes=num_classes,
                    max_nodes=max_nodes,
                    pool_ratio=args.pool_ratio,
                    dropout=args.dropout,
                ).to(device)

                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                )

                best_val = 0.0
                best_test = 0.0
                best_test_macro_f1 = 0.0
                epochs_no_improve = 0
                start_time = time.time()

                for epoch in range(1, args.epochs + 1):
                    epoch_start_time = time.perf_counter()
                    loss, epoch_train_time = train_one_epoch(model, train_loader, optimizer, device)
                    val_acc, val_macro_f1 = evaluate(model, val_loader, device)
                    test_acc, test_macro_f1 = evaluate(model, test_loader, device)
                    epoch_duration = time.perf_counter() - epoch_start_time

                    if val_acc > best_val + args.tolerance:
                        best_val = val_acc
                        best_test = test_acc
                        best_test_macro_f1 = test_macro_f1
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

                    logger.info(
                        f'Fold: {fold_idx}, Seed: {seed}, '
                        f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
                        f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
                        f'Epoch Time: {epoch_duration:.2f}s'
                    )

                    if args.early_stop > 0 and epochs_no_improve >= args.early_stop:
                        logger.info(f'Early stopping at epoch {epoch} for seed {seed}')
                        break

                total_time = time.time() - start_time
                memory_allocated = (
                    torch.cuda.memory_reserved(device) / (1024 ** 2)
                    if device.type == 'cuda' else 0.0
                )
                times.append(total_time)
                memories.append(memory_allocated)
                best_val_accs.append(best_val)
                best_test_accs.append(best_test)
                best_test_macro_f1s.append(best_test_macro_f1)

                if device.type == 'cuda':
                    torch.cuda.empty_cache()

        logger.info(f'Average Time: {np.mean(times):.2f} seconds')
        logger.info(f'Var Time: {np.var(times):.2f} seconds')
        logger.info(f'Average Memory: {np.mean(memories):.2f} MB')
        logger.info(f'Average Best Val Acc: {np.mean(best_val_accs):.4f}')
        logger.info(f'Average Accuracy/Micro-F1: {np.mean(best_test_accs):.4f}')
        logger.info(f'Std Accuracy/Micro-F1: {np.std(best_test_accs):.4f}')
        logger.info(f'Average Macro-F1: {np.mean(best_test_macro_f1s):.4f}')
        logger.info(f'Std Macro-F1: {np.std(best_test_macro_f1s):.4f}')

        csv_path = save_to_csv(
            args=args,
            model=method,
            timestamp=timestamp,
            times=times,
            memories=memories,
            max_nodes=max_nodes,
            best_val_accs=best_val_accs,
            best_test_accs=best_test_accs,
            best_test_macro_f1s=best_test_macro_f1s,
        )
        logger.info(f'Saved CSV: {csv_path}')


if __name__ == '__main__':
    main()
