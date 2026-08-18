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
from torch_geometric.loader import DataLoader

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


def evaluate_classification(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if batch.x is None or batch.x.size(1) == 0:
                x = torch.ones(
                    (batch.num_nodes, 1),
                    dtype=torch.float,
                    device=batch.edge_index.device,
                )
            else:
                x = batch.x

            output = model(x, batch.edge_index, batch.batch)
            predictions.append(output.argmax(dim=1).cpu())
            targets.append(batch.y.view(-1).cpu())

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

    dataset = TUDataset(root=data_dir, name=dataset_name)
    output_path = data_dir / f"{dataset_name}.pt"
    torch.save(dataset, output_path)

    is_dense = model_name in {
        "diff",
        "mincut",
        "gaus",
        "unif",
        "count1",
        "count2",
        "count4",
    }
    max_nodes = DENSE_MAX_NODES[dataset_name] if is_dense else None
    input_dim = max(1, dataset.num_features)
    num_classes = dataset.num_classes

    if is_dense:
        dataset = [data for data in dataset if data.num_nodes <= max_nodes]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    criterion = torch.nn.CrossEntropyLoss()
    indices = list(range(len(dataset)))
    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_splits = list(splitter.split(indices))
    fold_times = []
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
        train_val = [dataset[i] for i in train_val_indices]
        test_dataset = [dataset[i] for i in test_indices]
        random.shuffle(train_val)
        validation_size = max(1, int(0.1 * len(train_val)))
        validation_dataset = train_val[:validation_size]
        train_dataset = train_val[validation_size:]

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
            torch.cuda.synchronize(device)
        training_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            total_loss = 0.0
            total_pool_time = 0.0

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                if batch.x is None or batch.x.size(1) == 0:
                    x = torch.ones(
                        (batch.num_nodes, 1),
                        dtype=torch.float,
                        device=batch.edge_index.device,
                    )
                else:
                    x = batch.x
                output = model(x, batch.edge_index, batch.batch)
                auxiliary_loss = getattr(model, "last_auxiliary_loss", None)
                if auxiliary_loss is None:
                    auxiliary_loss = output.new_zeros(())
                loss = criterion(output, batch.y) + auxiliary_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_pool_time += model.last_pool_time

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            average_loss = total_loss / len(train_loader)
            epoch_duration = time.perf_counter() - epoch_start
            logger.info(
                f"Seed {seed}, Fold {fold} Epoch {epoch}/{epochs} "
                f"- loss: {average_loss:.4f} "
                f"- Epoch Time: {epoch_duration:.2f}s "
                f"- Pool Time: {total_pool_time:.2f}s"
            )

        fold_time = time.perf_counter() - training_start
        fold_times.append(fold_time)
        if device.type == "cuda":
            fold_memories.append(
                torch.cuda.memory_reserved(device) / (1024 ** 2)
            )

        validation_metrics = evaluate_classification(
            model, validation_loader, device
        )
        test_metrics = evaluate_classification(model, test_loader, device)
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

    total_training_time = sum(fold_times)
    logger.info(
        f"Total training time across {total_runs} fold/seed runs: "
        f"{total_training_time:.2f}s"
    )
    logger.info(f"Average training time per run: {sum(fold_times) / total_runs:.2f}s")

    average_metrics = {
        key: sum(metrics[key] for metrics in fold_metrics) / total_runs
        for key in fold_metrics[0]
    }
    logger.info(f"Average Accuracy: {average_metrics['accuracy']:.4f}")
    logger.info(f"Average Micro-F1: {average_metrics['micro_f1']:.4f}")
    logger.info(f"Average Macro-F1: {average_metrics['macro_f1']:.4f}")

    if device.type == "cuda":
        logger.info(
            f"Average GPU memory reserved: "
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
        times=fold_times,
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
