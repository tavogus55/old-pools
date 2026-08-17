import csv
import json
import logging
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomFormatter(logging.Formatter):
    """Custom formatter to include the current GPU in log messages with colors."""

    # ANSI color codes
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    # Log format with GPU info
    format = "%(asctime)s - %(gpu_info)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    # Different colors for different log levels
    FORMATS = {
        logging.DEBUG: green + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        # Get current GPU info
        if torch.cuda.is_available():
            gpu_id = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_id)
            record.gpu_info = f"GPU: {gpu_id} ({gpu_name})"
        else:
            record.gpu_info = "GPU: CPU"

        # Select the appropriate format based on log level
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_logger(exp_name, timestamp):
    """Sets up the logger with GPU info and color-coded formatting."""
    with open("global_settings.json", "r") as file:
        loaded_data = json.load(file)

    logger_settings = loaded_data["logger"]
    model = logger_settings["model"]
    dataset_name = logger_settings["dataset"]
    log_level = logger_settings["log_level"]

    # Build logs directory: logs/<exp_name>/
    logs_dir = os.path.join(os.getcwd(), "logs", exp_name)
    os.makedirs(logs_dir, exist_ok=True)

    # Build log filename: model_dataset_expname_timestamp.log
    filename = f"{model}_dataset-{dataset_name}_{exp_name}_{timestamp}.log"
    log_path = os.path.join(logs_dir, filename)

    logger = logging.getLogger(f"{model}_{dataset_name}_{exp_name}_{timestamp}")

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(CustomFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Set the logger level dynamically
        logger.setLevel(logging.DEBUG if log_level == "DEBUG" else logging.INFO)

    return logger


def log_experiment_settings(logger, args):
    """
    Logs all experiment settings in a nicely formatted table as a single log entry,
    including the PyTorch version.

    Args:
        logger: The logger instance to use.
        args: An argparse.Namespace or any object with attributes to log.
    """
    # Find the longest argument name for alignment
    max_len = max(len(k) for k in vars(args).keys())

    # Build the table as a string
    lines = []
    lines.append("-" * (max_len + 30))
    lines.append(f"{'Argument'.ljust(max_len)} | Value")
    lines.append("-" * (max_len + 30))

    for k, v in vars(args).items():
        lines.append(f"{k.ljust(max_len)} | {v}")

    # Add PyTorch version
    lines.append(f"{'torch_version'.ljust(max_len)} | {torch.__version__}")

    lines.append("-" * (max_len + 30))

    # Join everything into a single string and log once
    logger.info("Experiment settings:\n" + "\n".join(lines))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for batch in loader:
        batch = batch.to(device)
        logits, _ = model(batch)
        y_true.append(batch.y.view(-1).cpu())
        y_pred.append(logits.argmax(dim=-1).cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return acc, macro_f1


def save_to_csv(args, model, timestamp, times, memories, max_nodes,
                best_val_accs, best_test_accs, best_test_macro_f1s,
                save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{args.exp_name}.csv")

    row = {
        "timestamp": timestamp,
        "exp_name": args.exp_name,
        "model": model,
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "max_nodes": max_nodes,
        "k_folds": args.k_folds,
        "pool_ratio": args.pool_ratio,
        "early_stop": args.early_stop,
        "tolerance": args.tolerance,
        "seeds": "-".join(map(str, args.seeds)),
        "avg_time": np.mean(times),
        "var_time": np.var(times),
        "avg_memory_mb": np.mean(memories),
        "avg_best_val_acc": np.mean(best_val_accs),
        "std_best_val_acc": np.std(best_val_accs),
        "avg_test_acc_micro_f1": np.mean(best_test_accs),
        "std_test_acc_micro_f1": np.std(best_test_accs),
        "avg_macro_f1": np.mean(best_test_macro_f1s),
        "std_macro_f1": np.std(best_test_macro_f1s),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return csv_path
