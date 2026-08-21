import csv
import json
import logging
import os

import numpy as np
import torch


class CustomFormatter(logging.Formatter):
    """Logger formatter matching the new-pools implementation."""

    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    format = (
        "%(asctime)s - %(gpu_info)s - %(name)s - %(levelname)s - "
        "%(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            record.gpu_info = f"GPU: {gpu_id} ({gpu_name})"
        else:
            record.gpu_info = "GPU: CPU"

        formatter = logging.Formatter(self.FORMATS.get(record.levelno))
        return formatter.format(record)


def get_logger(exp_name, timestamp):
    """Set up the logger with GPU info and color-coded formatting."""
    with open("global_settings.json", "r") as file:
        loaded_data = json.load(file)

    logger_settings = loaded_data["logger"]
    model = logger_settings["model"]
    dataset_name = logger_settings["dataset"]
    log_level = logger_settings["log_level"]

    logs_dir = os.path.join(os.getcwd(), "logs", exp_name)
    os.makedirs(logs_dir, exist_ok=True)
    filename = f"{model}_dataset-{dataset_name}_{exp_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, filename)

    logger = logging.getLogger(f"{model}_{dataset_name}_{exp_name}_{timestamp}")
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

    return logger


def log_experiment_settings(logger, args):
    """Log all experiment arguments in the new-pools table format."""
    max_len = max(len(k) for k in vars(args).keys())
    lines = [
        "-" * (max_len + 30),
        f"{'Argument'.ljust(max_len)} | Value",
        "-" * (max_len + 30),
    ]
    for key, value in vars(args).items():
        lines.append(f"{key.ljust(max_len)} | {value}")
    lines.append(f"{'torch_version'.ljust(max_len)} | {torch.__version__}")
    lines.append("-" * (max_len + 30))
    logger.info("Experiment settings:\n" + "\n".join(lines))


def save_to_csv(
    args,
    task_type,
    timestamp,
    times,
    memories,
    max_nodes,
    best_val_accs=None,
    best_test_accs=None,
    best_test_macro_f1s=None,
    best_val_mses=None,
    best_test_mses=None,
    best_test_rmses=None,
    best_test_maes=None,
    end_to_end_times=None,
    training_times=None,
    evaluation_times=None,
    pool_times=None,
    preprocessing_times=None,
    save_dir="results",
):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{args.exp_name}.csv")
    # The primary time metric is the complete train + validation + test run.
    if end_to_end_times is None:
        end_to_end_times = times
    if training_times is None:
        training_times = times
    if evaluation_times is None:
        evaluation_times = [0.0]
    if pool_times is None:
        pool_times = [0.0]
    if preprocessing_times is None:
        preprocessing_times = [0.0]

    row = {
        "timestamp": timestamp,
        "exp_name": args.exp_name,
        "model": args.model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_nodes": max_nodes,
        "k_folds": args.k_folds,
        "pool_ratio": args.pratio,
        "early_stop": args.early_stop,
        "tolerance": args.tolerance,
        "seeds": "-".join(map(str, args.seeds)),
        "avg_time": np.mean(end_to_end_times),
        "var_time": np.var(end_to_end_times),
        "avg_training_time": np.mean(training_times),
        "avg_evaluation_time": np.mean(evaluation_times),
        "avg_pool_time": np.mean(pool_times),
        "avg_preprocessing_time": np.mean(preprocessing_times),
        "avg_memory_mb": np.mean(memories),
    }

    if task_type == "regression":
        row.update({
            "avg_best_val_mse": np.mean(best_val_mses),
            "std_best_val_mse": np.std(best_val_mses),
            "avg_test_mse": np.mean(best_test_mses),
            "std_test_mse": np.std(best_test_mses),
            "avg_test_rmse": np.mean(best_test_rmses),
            "std_test_rmse": np.std(best_test_rmses),
            "avg_test_mae": np.mean(best_test_maes),
            "std_test_mae": np.std(best_test_maes),
        })
    else:
        row.update({
            "avg_best_val_acc": np.mean(best_val_accs),
            "std_best_val_acc": np.std(best_val_accs),
            "avg_test_acc_micro_f1": np.mean(best_test_accs),
            "std_test_acc_micro_f1": np.std(best_test_accs),
            "avg_macro_f1": np.mean(best_test_macro_f1s),
            "std_macro_f1": np.std(best_test_macro_f1s),
        })

    fieldnames = list(row.keys())
    file_exists = os.path.exists(csv_path)

    if file_exists:
        with open(csv_path, "r", newline="") as file:
            reader = csv.DictReader(file)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []

        # Rewrite older result files when the timing schema changes, keeping
        # previous rows and leaving new timing columns blank for them.
        if existing_fieldnames != fieldnames:
            with open(csv_path, "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)
                writer.writerow(row)
            return

    with open(csv_path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
