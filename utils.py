import csv
import json
import logging
import os
import sys
import tempfile
import traceback

import numpy as np
import torch


RUN_KEY = ("exp_name", "dataset", "model", "seed", "fold")
RAW_FIELDS = [
    "timestamp", "exp_name", "model", "dataset", "task", "seed", "fold",
    "test_graphs", "end_to_end_time", "training_time", "training_time_per_epoch",
    "training_epochs", "evaluation_time", "inference_time", "throughput_graphs_per_sec",
    "pool_time", "training_gpu_memory_mb", "inference_gpu_memory_mb",
    "peak_gpu_memory_mb", "preprocessing_time",
] + [f"{split}_{metric}" for metric in ("accuracy", "micro_f1", "macro_f1", "mse", "rmse", "mae")
     for split in ("validation", "test")]
FAILURE_FIELDS = [*RUN_KEY, "timestamp", "exception_type", "error", "log_file"]


class ResultPersistenceError(RuntimeError):
    pass


def read_csv_rows(path):
    try:
        with open(path, newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            if any(None in row for row in rows):
                raise ValueError(f"Malformed CSV: {path}")
            return rows, reader.fieldnames or []
    except FileNotFoundError:
        return [], []
    except Exception as error:
        raise ResultPersistenceError(f"Could not read results from {path}: {error}") from error


def row_key(row, fields=RUN_KEY):
    return tuple(str(row.get(field, "")) for field in fields)


def write_csv_rows(path, rows, fieldnames):
    temporary_path = None
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", delete=False,
            dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp",
        ) as file:
            temporary_path = file.name
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except Exception as error:
        raise ResultPersistenceError(f"Could not persist results to {path}: {error}") from error
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)


def upsert_csv(path, rows, fieldnames, key_fields):
    existing, existing_fields = read_csv_rows(path)
    merged = {row_key(row, key_fields): row for row in existing}
    merged.update((row_key(row, key_fields), row) for row in rows)
    write_csv_rows(path, list(merged.values()), list(dict.fromkeys(existing_fields + fieldnames)))


class ExperimentProgress:
    def __init__(self, args, task_type, timestamp, logger, save_dir="results"):
        self.args = args
        self.task_type = task_type
        self.timestamp = timestamp
        self.logger = logger
        self.save_dir = save_dir
        self.raw_path = os.path.join(save_dir, f"{args.exp_name}_raw.csv")
        self.failed_path = os.path.join(save_dir, f"{args.exp_name}_failed.csv")
        fold_count = min(args.k_folds, args.folds_to_run) if args.folds_to_run > 0 else args.k_folds
        self.planned = [(seed, fold) for seed in dict.fromkeys(args.seeds)
                        for fold in range(1, fold_count + 1)]
        self.results = {}
        rows, fields = read_csv_rows(self.raw_path)
        retained = []
        for row in rows:
            if self.matches(row):
                try:
                    combination = (int(row["seed"]), int(row["fold"]))
                    numeric_fields = RAW_FIELDS[7:20] + (
                        [f"{split}_{metric}" for split in ("validation", "test")
                         for metric in ("mse", "rmse", "mae")]
                        if task_type == "regression" else
                        [f"{split}_{metric}" for split in ("validation", "test")
                         for metric in ("accuracy", "micro_f1", "macro_f1")]
                    )
                    complete = (row.get("task") == task_type and bool(row.get("timestamp"))
                                and all(np.isfinite(float(row[field])) for field in numeric_fields)
                                and float(row["training_epochs"]) > 0)
                except (KeyError, ValueError, TypeError):
                    complete = False
                if not complete:
                    continue
                if combination in self.planned:
                    self.results[combination] = row
            retained.append(row)
        retained = list({row_key(row): row for row in retained}.values())
        write_csv_rows(self.raw_path, retained, list(dict.fromkeys(fields + RAW_FIELDS)))
        self.reconcile_failures()

    def matches(self, row):
        return row_key(row, RUN_KEY[:3]) == (self.args.exp_name, self.args.dataset, self.args.model)

    def reconcile_failures(self):
        rows, fields = read_csv_rows(self.failed_path)
        completed_keys = {row_key(row) for row in self.results.values()}
        unresolved = {row_key(row): row for row in rows if row_key(row) not in completed_keys}
        write_csv_rows(self.failed_path, list(unresolved.values()),
                       list(dict.fromkeys(fields + FAILURE_FIELDS)))

    def fail(self, seed, fold, error):
        log_files = [handler.baseFilename for handler in self.logger.handlers
                     if isinstance(handler, logging.FileHandler)]
        row = dict(zip(RUN_KEY, (self.args.exp_name, self.args.dataset, self.args.model, seed, fold)))
        row.update(timestamp=self.timestamp, exception_type=type(error).__name__,
                   error=str(error), log_file=";".join(log_files))
        upsert_csv(self.failed_path, [row], FAILURE_FIELDS, RUN_KEY)

    def report_error(self, message):
        try:
            self.logger.exception(message)
        except Exception:
            traceback.print_exc()
            print(message, file=sys.stderr)

    def save(self, max_nodes):
        rows = list(self.results.values())

        def values(field):
            return [float(row[field]) for row in rows] or [float("nan")]

        metrics = ({
            "best_val_mses": values("validation_mse"),
            "best_test_mses": values("test_mse"),
            "best_test_rmses": values("test_rmse"),
            "best_test_maes": values("test_mae"),
        } if self.task_type == "regression" else {
            "best_val_accs": values("validation_accuracy"),
            "best_test_accs": values("test_accuracy"),
            "best_test_macro_f1s": values("test_macro_f1"),
        })
        save_to_csv(
            args=self.args, task_type=self.task_type, timestamp=self.timestamp,
            times=values("end_to_end_time"), memories=values("peak_gpu_memory_mb"),
            max_nodes=max_nodes, end_to_end_times=values("end_to_end_time"),
            training_times=values("training_time"), evaluation_times=values("evaluation_time"),
            inference_times=values("inference_time"), training_epoch_counts=values("training_epochs"),
            training_peak_memories=values("training_gpu_memory_mb"),
            inference_peak_memories=values("inference_gpu_memory_mb"),
            test_graph_counts=values("test_graphs"), pool_times=values("pool_time"),
            preprocessing_times=values("preprocessing_time"), raw_results=rows,
            save_dir=self.save_dir, **metrics,
        )
        self.reconcile_failures()


class ResultFileHandler(logging.FileHandler):
    def handleError(self, record):
        print("Experiment logging failed; falling back to stderr:", file=sys.stderr)
        traceback.print_exc()
        print(logging.Formatter().format(record), file=sys.stderr)


class ResultStreamHandler(logging.StreamHandler):
    handleError = ResultFileHandler.handleError


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
        file_handler = ResultFileHandler(log_path)
        file_handler.setFormatter(CustomFormatter())
        console_handler = ResultStreamHandler()
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
    inference_times=None,
    training_epoch_counts=None,
    training_peak_memories=None,
    inference_peak_memories=None,
    test_graph_counts=None,
    pool_times=None,
    preprocessing_times=None,
    raw_results=None,
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
    if inference_times is None:
        inference_times = evaluation_times
    if training_epoch_counts is None:
        training_epoch_counts = [args.epochs] * len(training_times)
    if training_peak_memories is None:
        training_peak_memories = [0.0]
    if inference_peak_memories is None:
        inference_peak_memories = [0.0]
    if test_graph_counts is None:
        test_graph_counts = [0] * len(inference_times)
    if pool_times is None:
        pool_times = [0.0]
    if preprocessing_times is None:
        preprocessing_times = [0.0]

    def std(values):
        # Use sample standard deviation for mean +/- std reporting.
        return np.std(values, ddof=1) if len(values) > 1 else 0.0

    training_time_per_epoch = [
        training_time / max(epochs, 1)
        for training_time, epochs in zip(training_times, training_epoch_counts)
    ]

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
        "mp_layer": args.mp_layer,
        "max_nodes": max_nodes,
        "k_folds": args.k_folds,
        "pool_ratio": args.pratio,
        "early_stop": args.early_stop,
        "seeds": "-".join(map(str, args.seeds)),
        "avg_time": np.mean(end_to_end_times),
        "std_time": std(end_to_end_times),
        "var_time": np.var(end_to_end_times),
        "avg_training_time": np.mean(training_times),
        "std_training_time": std(training_times),
        "avg_training_time_per_epoch": np.mean(training_time_per_epoch),
        "std_training_time_per_epoch": std(training_time_per_epoch),
        "avg_training_epochs": np.mean(training_epoch_counts),
        "std_training_epochs": std(training_epoch_counts),
        "total_training_epochs": np.sum(training_epoch_counts),
        "total_training_time": np.sum(training_times),
        "avg_evaluation_time": np.mean(evaluation_times),
        "std_evaluation_time": std(evaluation_times),
        "avg_inference_time": np.mean(inference_times),
        "std_inference_time": std(inference_times),
        "total_inference_time": np.sum(inference_times),
        "avg_throughput_graphs_per_sec": np.mean(
            np.asarray(test_graph_counts) / np.maximum(np.asarray(inference_times), 1e-12)
        ),
        "std_throughput_graphs_per_sec": std(
            np.asarray(test_graph_counts) / np.maximum(np.asarray(inference_times), 1e-12)
        ),
        "test_throughput_graphs_per_sec": (
            np.sum(test_graph_counts) / max(np.sum(inference_times), 1e-12)
        ),
        "avg_pool_time": np.mean(pool_times),
        "std_pool_time": std(pool_times),
        "avg_preprocessing_time": np.mean(preprocessing_times),
        "avg_memory_mb": np.mean(memories),
        "std_memory_mb": std(memories),
        "avg_training_gpu_memory_mb": np.mean(training_peak_memories),
        "std_training_gpu_memory_mb": std(training_peak_memories),
        "peak_training_gpu_memory_mb": np.max(training_peak_memories),
        "avg_inference_gpu_memory_mb": np.mean(inference_peak_memories),
        "std_inference_gpu_memory_mb": std(inference_peak_memories),
        "peak_inference_gpu_memory_mb": np.max(inference_peak_memories),
    }

    if task_type == "regression":
        row.update({
            "avg_best_val_mse": np.mean(best_val_mses),
            "std_best_val_mse": std(best_val_mses),
            "avg_test_mse": np.mean(best_test_mses),
            "std_test_mse": std(best_test_mses),
            "avg_test_rmse": np.mean(best_test_rmses),
            "std_test_rmse": std(best_test_rmses),
            "avg_test_mae": np.mean(best_test_maes),
            "std_test_mae": std(best_test_maes),
        })
    else:
        row.update({
            "avg_best_val_acc": np.mean(best_val_accs),
            "std_best_val_acc": std(best_val_accs),
            "avg_test_acc_micro_f1": np.mean(best_test_accs),
            "std_test_acc_micro_f1": std(best_test_accs),
            "avg_macro_f1": np.mean(best_test_macro_f1s),
            "std_macro_f1": std(best_test_macro_f1s),
        })

    if raw_results is not None:
        row["planned_runs"] = len(set(args.seeds)) * (
            min(args.k_folds, args.folds_to_run)
            if args.folds_to_run > 0 else args.k_folds
        )
        row["successful_runs"] = len(raw_results)
        if not raw_results:
            for field in row:
                if field.startswith(("avg_", "std_", "var_", "peak_", "test_throughput")):
                    row[field] = ""
                elif field.startswith("total_"):
                    row[field] = 0
        raw_csv_path = os.path.join(save_dir, f"{args.exp_name}_raw.csv")
        upsert_csv(raw_csv_path, raw_results, RAW_FIELDS, RUN_KEY)
    upsert_csv(csv_path, [row], list(row), ("exp_name", "dataset", "model"))
