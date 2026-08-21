import argparse
import json
from pathlib import Path
import random
import time
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader, DenseDataLoader
from torch_geometric.transforms import ToDense

from pooling_models import DensePool, sparse_pooling
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
        x = batch.x

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
    k_folds: int,
    seeds,
    logger,
    args,
    timestamp: str,
) -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"

    preprocessing_start = time.perf_counter()
    dataset = TUDataset(root=data_dir, name=dataset_name)
    output_path = data_dir / f"{dataset_name}.pt"
    input_dim = max(1, dataset.num_features)
    num_classes = dataset.num_classes

    is_dense = model_name in {
        "diff",
        "mincut",
        "gaus",
        "unif",
        "count1",
        "count2",
        "count4",
    }
    # Apply the same graph-size limit to every model so sparse and dense
    # methods process exactly the same dataset and cross-validation splits.
    max_nodes = DENSE_MAX_NODES[dataset_name]
    original_graph_count = len(dataset)
    dataset = [data for data in dataset if data.num_nodes <= max_nodes]
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
            # DenseGCNConv expects a 2D adjacency matrix per graph, so do not
            # let MUTAG edge features create an extra adjacency dimension.
            if getattr(dense_data, "edge_attr", None) is not None:
                dense_data.edge_attr = None
            if dense_data.x is None or dense_data.x.size(1) == 0:
                dense_data.x = torch.ones((dense_data.num_nodes, 1))
            dense_data = to_dense(dense_data)
            if dense_data.adj.dim() == 3:
                dense_data.adj = (
                    dense_data.adj.abs().sum(dim=-1) > 0
                ).float()
            dense_dataset.append(dense_data)

    preprocessing_time = time.perf_counter() - preprocessing_start
    logger.info(f"Preprocessing time: {preprocessing_time:.2f}s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    criterion = torch.nn.CrossEntropyLoss()
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

        # Start the end-to-end run clock before model construction.
        run_start = synchronized_time(device)

        if is_dense:
            dense_train_dataset = [dense_dataset[i] for i in train_indices]
            dense_validation_dataset = [dense_dataset[i] for i in validation_indices]
            dense_test_dataset = [dense_dataset[i] for i in test_indices]
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
                train_dataset, batch_size=batch_size, shuffle=True
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
            ).to(device)
        else:
            model = sparse_pooling(
                input_dim,
                num_classes,
                model=model_name,
                hidden=hidden,
                pratio=pratio,
                dropout=dropout,
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

        for epoch in range(1, epochs + 1):
            model.reset_pool_timing()
            epoch_start = synchronized_time(device)
            total_loss = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                output = forward_model(model, batch, device, is_dense)
                auxiliary_loss = getattr(model, "last_auxiliary_loss", None)
                if auxiliary_loss is None:
                    auxiliary_loss = output.new_zeros(())
                target = batch.y.view(-1)
                loss = criterion(output, target) + auxiliary_loss
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

        training_time = synchronized_time(device) - training_start
        training_times.append(training_time)
        evaluation_start = synchronized_time(device)

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

    logger.info(f"Loaded {len(dataset)} graphs from the {dataset_name} TU dataset.")
    logger.info(f"Saved dataset to {output_path}")

    save_to_csv(
        args=args,
        task_type="multiclass",
        timestamp=timestamp,
        times=end_to_end_times,
        memories=fold_memories if fold_memories else [0.0],
        max_nodes=max_nodes,
        best_val_accs=[
            metrics["accuracy"] for metrics in validation_fold_metrics
        ],
        best_test_accs=[
            metrics["accuracy"] for metrics in fold_metrics
        ],
        best_test_macro_f1s=[
            metrics["macro_f1"] for metrics in fold_metrics
        ],
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
            "diff",
            "mincut",
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
        args.k_folds,
        args.seeds,
        logger,
        args,
        timestamp,
    )
