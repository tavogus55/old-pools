import argparse
import json
from pathlib import Path
import random
import time
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch_geometric.datasets import MoleculeNet, QM7b, TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.transforms import ToDense

from dataset_loader import CSVMoleculeDataset
from pooling_models import CountSketchPooling, DensePool, sparse_pooling
from utils import get_logger, log_experiment_settings, save_to_csv


DENSE_MAX_NODES = {
    "MUTAG": 150,
    "DD": 500,
    "IMDB-MULTI": 500,
    "PROTEINS": 700,
    "IMDB-BINARY": 500,
    "COLLAB": 150,
    "NCI1": 150,
    "NCI109": 150,
    "ESOL": 64,
    "FreeSolv": 32,
    "lipo": 128,
    "QM7": 32,
    "QM8": 32,
    "BACE": 96,
    "QM7b": 32,
    "ENZYMES": 126,
    "PTC_MR": 109,
    "AIDS": 200,
    "MUTAGENICITY": 500,
    "REDDIT-BINARY": 500,
    "REDDIT-MULTI-5K": 500,
    "BZR": 100,
    "COX2": 100,
    "DHFR": 100,
    "MSRC_9": 100,
    "MSRC_21": 100,
    "COIL-DEL": 100,
    "Synthie": 100,
}

REGRESSION_DATASETS = {
    "QM7",
    "QM8",
    "BACE",
    "ESOL",
    "FreeSolv",
    "lipo",
    "QM7b",
}


def forward_model(model, batch, device, is_dense):
    """Run a model using either sparse or precomputed dense batch data."""
    # Dense pooling models use precomputed features, adjacency matrices, and masks.
    if is_dense:
        return model(
            batch.x,
            adj=batch.adj,
            mask=batch.mask,
        )

    # Featureless datasets (IMDB-MULTI, IMDB-BINARY, and COLLAB) receive constant features.
    if batch.x is None or batch.x.size(1) == 0:
        x = torch.ones(
            (batch.num_nodes, 1),
            dtype=torch.float,
            device=batch.edge_index.device,
        )
    # Feature-aware datasets (PROTEINS, DD, MUTAG, NCI1, and NCI109) use supplied features.
    else:
        # MoleculeNet atom features may be stored as integer tensors, while
        # all message-passing layers operate on floating-point features.
        x = batch.x.float()

    # CountSketch accepts optional scalar edge weights; multidimensional TU
    # edge attributes are not valid scalar weights and therefore default to 1.
    if isinstance(model, CountSketchPooling):
        edge_weight = getattr(batch, "edge_attr", None)
        if edge_weight is not None and edge_weight.dim() != 1:
            edge_weight = None
        return model(
            x,
            batch.edge_index,
            batch.batch,
            edge_weight=edge_weight,
        )

    return model(x, batch.edge_index, batch.batch)


def evaluate_classification(model, loader, device, is_dense):
    model.eval()
    model.reset_pool_timing()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = forward_model(model, batch, device, is_dense)
            predictions.append(output.argmax(dim=1).cpu())
            targets.append(batch.y.view(-1).cpu())

    # Resolve pooling events once for the complete evaluation pass.
    model.finish_pool_timing()

    predictions = torch.cat(predictions).numpy()
    targets = torch.cat(targets).numpy()
    return {
        "accuracy": accuracy_score(targets, predictions),
        "micro_f1": f1_score(
            targets,
            predictions,
            average="micro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronized_time(device):
    """Return a wall-clock timestamp after all CUDA work has completed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def evaluate_regression(model, loader, device, is_dense, target_mean, target_std):
    """Evaluate regression predictions after returning them to target units."""
    model.eval()
    model.reset_pool_timing()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = forward_model(model, batch, device, is_dense)
            target = batch.y.float()
            output = output * target_std + target_mean
            predictions.append(output.view(-1).cpu())
            targets.append(target.view(-1).cpu())

    model.finish_pool_timing()
    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    error = predictions - targets
    mse = torch.mean(error.pow(2)).item()
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": torch.mean(error.abs()).item(),
    }


def normalize_regression_dataset(dataset, mean, std):
    """Clone graphs and normalize only their regression targets."""
    normalized = []
    for data in dataset:
        item = data.clone()
        item.y = (item.y.float() - mean) / std
        normalized.append(item)
    return normalized


def main(
    dataset_name: str,
    model_name: str,
    epochs: int,
    hidden: int,
    pratio: float,
    learning_rate: float,
    weight_decay: float,
    dropout: float,
    batch_size: int,
    mp_layer: str,
    k_folds: int,
    seeds,
    logger,
    args,
    timestamp: str,
) -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    preprocessing_start = time.perf_counter()
    task_type = "regression" if dataset_name in REGRESSION_DATASETS else "multiclass"
    if dataset_name in {"QM7", "QM8", "BACE"}:
        target_cols = ["pIC50"] if dataset_name == "BACE" else None
        dataset = CSVMoleculeDataset(
            root=data_dir / dataset_name,
            csv_file=data_dir / dataset_name / f"{dataset_name.lower()}.csv",
            target_cols=target_cols,
        )
    elif dataset_name == "QM7b":
        dataset = QM7b(root=data_dir / dataset_name)
    elif task_type == "regression":
        dataset = MoleculeNet(root=data_dir, name=dataset_name)
    else:
        dataset = TUDataset(root=data_dir, name=dataset_name)
    output_path = data_dir / f"{dataset_name}.pt"
    input_dim = max(1, dataset.num_features)
    target_dim = int(dataset[0].y.numel())
    num_classes = dataset.num_classes if task_type != "regression" else target_dim

    is_dense = model_name in {
        "diff",
        "mincut",
        "gaus",
        "unif",
        "dmon",
        "hosc",
        "justb",
    }
    # Apply the same graph-size limit to every model so sparse and dense
    # methods process exactly the same dataset and cross-validation splits.
    max_nodes = DENSE_MAX_NODES[dataset_name]
    original_graph_count = len(dataset)
    dataset = [data for data in dataset if data.num_nodes <= max_nodes]
    # Keep graph-level regression targets independent of node padding.
    if task_type == "regression":
        for data in dataset:
            data.y = data.y.float().view(-1)
    logger.info(
        f"Max-node filter: kept {len(dataset)}/{original_graph_count} graphs "
        f"with at most {max_nodes} nodes"
    )

    # Save the filtered dataset used by the experiment.
    torch.save(dataset, output_path)
    dense_dataset = None
    if is_dense:
        # Convert dense inputs once before training instead of rebuilding
        # padded features and adjacency matrices inside every forward pass.
        to_dense = ToDense(max_nodes)
        dense_dataset = []
        for data in dataset:
            dense_data = data.clone()
            # ToDense may pad graph attributes such as y; preserve the target
            # separately because regression labels are graph-level values.
            graph_target = dense_data.y.clone()
            # DenseGCNConv expects a 2D adjacency matrix per graph, so do not
            # let MUTAG edge features create an extra adjacency dimension.
            if getattr(dense_data, "edge_attr", None) is not None:
                dense_data.edge_attr = None
            if dense_data.x is None or dense_data.x.size(1) == 0:
                dense_data.x = torch.ones((dense_data.num_nodes, 1))
            else:
                dense_data.x = dense_data.x.float()
            dense_data = to_dense(dense_data)
            dense_data.y = graph_target
            if dense_data.adj.dim() == 3:
                dense_data.adj = (
                    dense_data.adj.abs().sum(dim=-1) > 0
                ).float()
            dense_dataset.append(dense_data)

    preprocessing_time = time.perf_counter() - preprocessing_start
    logger.info(f"Preprocessing time: {preprocessing_time:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    criterion = torch.nn.MSELoss() if task_type == "regression" else torch.nn.CrossEntropyLoss()
    indices = list(range(len(dataset)))
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_splits = list(splitter.split(indices))
    end_to_end_times = []
    training_times = []
    evaluation_times = []
    pool_times = []
    fold_memories = []
    fold_metrics = []
    validation_fold_metrics = []

    total_runs = len(seeds) * k_folds
    for run, (seed, (train_val_indices, test_indices)) in enumerate(
        (
            (seed, split)
            for seed in seeds
            for split in fold_splits
        ),
        start=1,
    ):
        fold = ((run - 1) % k_folds) + 1
        set_seed(seed)
        train_val_indices = list(train_val_indices)
        random.shuffle(train_val_indices)
        validation_size = max(1, int(0.1 * len(train_val_indices)))
        validation_indices = train_val_indices[:validation_size]
        train_indices = train_val_indices[validation_size:]

        train_dataset = [dataset[i] for i in train_indices]
        validation_dataset = [dataset[i] for i in validation_indices]
        test_dataset = [dataset[i] for i in test_indices]

        if task_type == "regression":
            # Compute normalization statistics from the training split only.
            train_targets = torch.cat(
                [data.y.view(1, -1).float() for data in train_dataset], dim=0
            )
            target_mean = train_targets.mean(dim=0).to(device)
            target_std = train_targets.std(dim=0, unbiased=False).clamp_min(1e-8).to(device)
            normalized_train_dataset = normalize_regression_dataset(
                train_dataset, target_mean.cpu(), target_std.cpu()
            )
        else:
            target_mean = target_std = None
            normalized_train_dataset = train_dataset

        # Start the end-to-end run clock before model construction.
        run_start = synchronized_time(device)

        if is_dense:
            dense_train_dataset = [dense_dataset[i] for i in train_indices]
            dense_validation_dataset = [dense_dataset[i] for i in validation_indices]
            dense_test_dataset = [dense_dataset[i] for i in test_indices]
            if task_type == "regression":
                dense_train_dataset = normalize_regression_dataset(
                    dense_train_dataset, target_mean.cpu(), target_std.cpu()
                )
            train_loader = DenseDataLoader(
                dense_train_dataset, batch_size=batch_size, shuffle=True
            )
            validation_loader = DenseDataLoader(
                dense_validation_dataset, batch_size=batch_size, shuffle=False
            )
            test_loader = DenseDataLoader(
                dense_test_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            train_loader = DataLoader(
                normalized_train_dataset, batch_size=batch_size, shuffle=True
            )
            validation_loader = DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

        if is_dense:
            model = DensePool(
                input_dim,
                num_classes,
                model=model_name,
                hidden=hidden,
                pratio=pratio,
                dropout=dropout,
                max_nodes=max_nodes,
                output_dim=target_dim if task_type == "regression" else None,
            ).to(device)
        elif model_name in {"count1", "count2", "count4"}:
            model = CountSketchPooling(
                input_dim,
                num_classes,
                q=int(model_name.removeprefix("count")),
                hidden=hidden,
                pratio=pratio,
                dropout=dropout,
                mp_layer=mp_layer,
                output_dim=target_dim if task_type == "regression" else None,
            ).to(device)
        else:
            model = sparse_pooling(
                input_dim,
                num_classes,
                model=model_name,
                hidden=hidden,
                pratio=pratio,
                dropout=dropout,
                mp_layer=mp_layer,
                output_dim=target_dim if task_type == "regression" else None,
            ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        logger.info(
            f"Seed {seed}, Fold {fold}/{k_folds}: "
            f"train={len(train_dataset)}, "
            f"validation={len(validation_dataset)}, test={len(test_dataset)}"
        )

        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        training_start = synchronized_time(device)
        run_pool_time = 0.0
        best_val_mse = float("inf")
        best_state = None
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            model.reset_pool_timing()
            epoch_start = synchronized_time(device)
            total_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = forward_model(model, batch, device, is_dense)
                if task_type == "regression":
                    target = batch.y.float()
                else:
                    target = batch.y.view(-1)
                    # Match the benchmark paper: optimize classification loss only.
                    # Pooling auxiliary objectives remain available on the model for
                    # diagnostics but are not added to the training objective.
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Resolve all pooling events once per epoch, rather than once per batch.
            model.finish_pool_timing()
            run_pool_time += model.last_pool_time
            average_loss = total_loss / len(train_loader)
            epoch_duration = time.perf_counter() - epoch_start
            logger.info(
                f"Seed {seed}, Fold {fold} Epoch {epoch}/{epochs} "
                f"- loss: {average_loss:.4f} "
                f"- Epoch Time: {epoch_duration:.2f}s "
                f"- Pool Time: {model.last_pool_time:.2f}s"
            )

            if task_type == "regression":
                epoch_validation = evaluate_regression(
                    model, validation_loader, device, is_dense,
                    target_mean, target_std
                )
                if epoch_validation["mse"] < best_val_mse - args.tolerance:
                    best_val_mse = epoch_validation["mse"]
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.early_stop:
                        logger.info(
                            f"Seed {seed}, Fold {fold}: early stopping at epoch {epoch}"
                        )
                        break

        training_time = synchronized_time(device) - training_start
        training_times.append(training_time)
        evaluation_start = synchronized_time(device)

        if task_type == "regression":
            if best_state is not None:
                model.load_state_dict(best_state)
            validation_metrics = evaluate_regression(
                model, validation_loader, device, is_dense,
                target_mean, target_std
            )
            test_metrics = evaluate_regression(
                model, test_loader, device, is_dense,
                target_mean, target_std
            )
        else:
            validation_metrics = evaluate_classification(
                model, validation_loader, device, is_dense
            )
            test_metrics = evaluate_classification(
                model, test_loader, device, is_dense
            )
        evaluation_time = synchronized_time(device) - evaluation_start
        evaluation_times.append(evaluation_time)
        end_to_end_time = synchronized_time(device) - run_start
        end_to_end_times.append(end_to_end_time)
        pool_times.append(run_pool_time)
        if device.type == "cuda":
            fold_memories.append(
                torch.cuda.max_memory_reserved(device) / (1024 ** 2)
            )
        logger.info(
            f"Seed {seed}, Fold {fold} timing - "
            f"End-to-end: {end_to_end_time:.2f}s, "
            f"Training: {training_time:.2f}s, "
            f"Evaluation: {evaluation_time:.2f}s"
        )
        fold_metrics.append(test_metrics)
        validation_fold_metrics.append(validation_metrics)
        if task_type == "regression":
            logger.info(
                f"Seed {seed}, Fold {fold} validation - "
                f"MSE: {validation_metrics['mse']:.4f}, "
                f"RMSE: {validation_metrics['rmse']:.4f}, "
                f"MAE: {validation_metrics['mae']:.4f}"
            )
            logger.info(
                f"Seed {seed}, Fold {fold} test - "
                f"MSE: {test_metrics['mse']:.4f}, "
                f"RMSE: {test_metrics['rmse']:.4f}, "
                f"MAE: {test_metrics['mae']:.4f}"
            )
        else:
            logger.info(
                f"Seed {seed}, Fold {fold} validation - "
                f"Accuracy: {validation_metrics['accuracy']:.4f}, "
                f"Micro-F1: {validation_metrics['micro_f1']:.4f}, "
                f"Macro-F1: {validation_metrics['macro_f1']:.4f}"
            )
            logger.info(
                f"Seed {seed}, Fold {fold} test - "
                f"Accuracy: {test_metrics['accuracy']:.4f}, "
                f"Micro-F1: {test_metrics['micro_f1']:.4f}, "
                f"Macro-F1: {test_metrics['macro_f1']:.4f}"
            )

    total_end_to_end_time = sum(end_to_end_times)
    logger.info(
        f"Total end-to-end time across {total_runs} fold/seed runs: "
        f"{total_end_to_end_time:.2f}s"
    )
    logger.info(
        f"Average end-to-end time per run: "
        f"{sum(end_to_end_times) / total_runs:.2f}s"
    )
    logger.info(
        f"Average training time per run: "
        f"{sum(training_times) / total_runs:.2f}s"
    )
    logger.info(
        f"Average evaluation time per run: "
        f"{sum(evaluation_times) / total_runs:.2f}s"
    )
    logger.info(
        f"Average pooling time per run: "
        f"{sum(pool_times) / total_runs:.2f}s"
    )

    average_metrics = {
        key: sum(metrics[key] for metrics in fold_metrics) / total_runs
        for key in fold_metrics[0]
    }
    if task_type == "regression":
        logger.info(f"Average MSE: {average_metrics['mse']:.4f}")
        logger.info(f"Average RMSE: {average_metrics['rmse']:.4f}")
        logger.info(f"Average MAE: {average_metrics['mae']:.4f}")
    else:
        logger.info(f"Average Accuracy: {average_metrics['accuracy']:.4f}")
        logger.info(f"Average Micro-F1: {average_metrics['micro_f1']:.4f}")
        logger.info(f"Average Macro-F1: {average_metrics['macro_f1']:.4f}")

    if device.type == "cuda":
        logger.info(
            f"Average peak GPU memory reserved: "
            f"{sum(fold_memories) / len(fold_memories):.2f} MB"
        )
    else:
        logger.info("GPU usage: unavailable (running on CPU)")

    logger.info(f"Loaded {len(dataset)} graphs from the {dataset_name} dataset.")
    logger.info(f"Saved dataset to {output_path}")

    if task_type == "regression":
        save_to_csv(
            args=args,
            task_type="regression",
            timestamp=timestamp,
            times=end_to_end_times,
            memories=fold_memories if fold_memories else [0.0],
            max_nodes=max_nodes,
            best_val_mses=[metrics["mse"] for metrics in validation_fold_metrics],
            best_test_mses=[metrics["mse"] for metrics in fold_metrics],
            best_test_rmses=[metrics["rmse"] for metrics in fold_metrics],
            best_test_maes=[metrics["mae"] for metrics in fold_metrics],
            end_to_end_times=end_to_end_times,
            training_times=training_times,
            evaluation_times=evaluation_times,
            pool_times=pool_times,
            preprocessing_times=[preprocessing_time],
        )
    else:
        save_to_csv(
            args=args,
            task_type="multiclass",
            timestamp=timestamp,
            times=end_to_end_times,
            memories=fold_memories if fold_memories else [0.0],
            max_nodes=max_nodes,
            best_val_accs=[metrics["accuracy"] for metrics in validation_fold_metrics],
            best_test_accs=[metrics["accuracy"] for metrics in fold_metrics],
            best_test_macro_f1s=[metrics["macro_f1"] for metrics in fold_metrics],
            end_to_end_times=end_to_end_times,
            training_times=training_times,
            evaluation_times=evaluation_times,
            pool_times=pool_times,
            preprocessing_times=[preprocessing_time],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=(
            "PROTEINS",
            "DD",
            "IMDB-MULTI",
            "IMDB-BINARY",
            "MUTAG",
            "NCI1",
            "NCI109",
            "COLLAB",
            "ESOL",
            "FreeSolv",
            "lipo",
            "QM7",
            "QM8",
            "BACE",
            "QM7b",
            "ENZYMES",
            "PTC_MR",
            "AIDS",
            "MUTAGENICITY",
            "REDDIT-BINARY",
            "REDDIT-MULTI-5K",
            "BZR",
            "COX2",
            "DHFR",
            "MSRC_9",
            "MSRC_21",
            "COIL-DEL",
            "Synthie",
        ),
        default="DD",
        help="TU dataset to use.",
    )
    parser.add_argument(
        "--model",
        choices=(
            "sag",
            "topk",
            "ndrp",
            "ndp",
            "graclus",
            "asapool",
            "pan",
            "cop",
            "cgi",
            "kmis",
            "gsap",
            "hgpsl",
            "hdpsl",
            "pars",
            "diff",
            "mincut",
            "dmon",
            "hosc",
            "justb",
            "gaus",
            "unif",
            "count1",
            "count2",
            "count4",
        ),
        default="topk",
        help="Pooling layer to use.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=32,
        help="Node embedding hidden dimension.",
    )
    parser.add_argument(
        "--pratio",
        type=float,
        default=0.5,
        help="Pooling ratio.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        type=float,
        default=1e-3,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-4,
        help="Optimizer weight decay.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout probability.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size.",
    )
    parser.add_argument(
        "--mp-layer",
        choices=("gcn", "graphconv"),
        default="graphconv",
        help="Sparse message-passing layer used by all sparse models.",
    )
    parser.add_argument(
        "--k_folds",
        "--k-folds",
        dest="k_folds",
        type=int,
        default=10,
        help="Number of shuffled cross-validation folds.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        nargs="+",
        default=[str(seed) for seed in range(42, 52)],
        help="Random seeds to run for every fold.",
    )
    parser.add_argument(
        "--log_level",
        choices=("debug", "info", "warning", "error", "critical"),
        default="info",
    )
    parser.add_argument("--log_path", default="local")
    parser.add_argument("--exp_name", default="exp")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--ddp", action="store_true")
    args = parser.parse_args()
    args.seeds = [
        int(seed)
        for seed_group in args.seeds
        for seed in seed_group.split(",")
        if seed
    ]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logger_settings = {
        "logger": {
            "model": args.model,
            "log_path": args.log_path,
            "dataset": args.dataset,
            "log_level": args.log_level.upper(),
        }
    }
    with open("global_settings.json", "w") as file:
        json.dump(logger_settings, file, indent=4)

    logger = get_logger(args.exp_name, timestamp)
    log_experiment_settings(logger, args)
    main(
        args.dataset,
        args.model,
        args.epochs,
        args.hidden,
        args.pratio,
        args.lr,
        args.weight_decay,
        args.dropout,
        args.batch_size,
        args.mp_layer,
        args.k_folds,
        args.seeds,
        logger,
        args,
        timestamp,
    )
