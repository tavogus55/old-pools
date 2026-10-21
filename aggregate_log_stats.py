"""Aggregate per-seed/fold metrics from old-pools experiment logs."""

import argparse
import csv
import re
from pathlib import Path
from statistics import mean, stdev


SETTING_RE = re.compile(r"^(\w+)\s+\|\s+(.+)$")
TIMING_RE = re.compile(
    r"Seed\s+(\d+),\s+Fold\s+(\d+)\s+timing\s+-\s+"
    r"End-to-end:\s+([0-9.eE+-]+)s,\s+"
    r"Training:\s+([0-9.eE+-]+)s,\s+"
    r"Evaluation:\s+([0-9.eE+-]+)s,\s+"
    r"Test inference:\s+([0-9.eE+-]+)s,\s+"
    r"Training epochs:\s+(\d+),\s+"
    r"Throughput:\s+([0-9.eE+-]+)\s+graphs/s"
)
CLASSIFICATION_RE = re.compile(
    r"Seed\s+(\d+),\s+Fold\s+(\d+)\s+test\s+-\s+"
    r"Accuracy:\s+([0-9.eE+-]+),\s+"
    r"Micro-F1:\s+([0-9.eE+-]+),\s+"
    r"Macro-F1:\s+([0-9.eE+-]+)"
)
REGRESSION_RE = re.compile(
    r"Seed\s+(\d+),\s+Fold\s+(\d+)\s+test\s+-\s+"
    r"MSE:\s+([0-9.eE+-]+),\s+"
    r"RMSE:\s+([0-9.eE+-]+),\s+"
    r"MAE:\s+([0-9.eE+-]+)"
)


def summarize(values):
    if not values:
        return "", ""
    average = mean(values)
    deviation = stdev(values) if len(values) > 1 else ""
    return f"{average:.6f}", "" if deviation == "" else f"{deviation:.6f}"


def parse_log(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    settings = {}
    for line in lines:
        match = SETTING_RE.match(line)
        if match:
            settings[match.group(1)] = match.group(2).strip()

    runs = {}

    def get_run(seed, fold):
        return runs.setdefault((int(seed), int(fold)), {})

    for line in lines:
        match = TIMING_RE.search(line)
        if match:
            run = get_run(match.group(1), match.group(2))
            run.update(
                end_to_end=float(match.group(3)),
                training=float(match.group(4)),
                evaluation=float(match.group(5)),
                inference=float(match.group(6)),
                epochs=float(match.group(7)),
                throughput=float(match.group(8)),
            )
            continue

        match = CLASSIFICATION_RE.search(line)
        if match:
            run = get_run(match.group(1), match.group(2))
            run.update(
                accuracy=float(match.group(3)),
                micro_f1=float(match.group(4)),
                macro_f1=float(match.group(5)),
            )
            continue

        match = REGRESSION_RE.search(line)
        if match:
            run = get_run(match.group(1), match.group(2))
            run.update(
                mse=float(match.group(3)),
                rmse=float(match.group(4)),
                mae=float(match.group(5)),
            )

    metric_names = [
        "accuracy", "micro_f1", "macro_f1", "mse", "rmse", "mae",
        "end_to_end", "training", "evaluation", "inference", "epochs",
        "throughput",
    ]
    row = {
        "log_file": str(path.resolve()),
        "log_timestamp": (
            re.search(r"_(\d{8}-\d{6})\.log$", path.name).group(1)
            if re.search(r"_(\d{8}-\d{6})\.log$", path.name)
            else ""
        ),
        "model": settings.get("model", ""),
        "dataset": settings.get("dataset", ""),
        "exp_name": settings.get("exp_name", ""),
        "completed_runs": len(runs),
    }

    try:
        fold_count = int(settings.get("k_folds", "0"))
        seed_count = len([s for s in re.findall(r"\d+", settings.get("seeds", ""))])
        expected_runs = fold_count * seed_count
    except ValueError:
        expected_runs = 0

    row["expected_runs"] = expected_runs
    row["complete"] = (
        "unknown" if expected_runs == 0 else str(len(runs) == expected_runs)
    )
    row["task"] = (
        "classification" if any("accuracy" in run for run in runs.values())
        else "regression" if any("mse" in run for run in runs.values())
        else "unknown"
    )

    for metric in metric_names:
        values = [run[metric] for run in runs.values() if metric in run]
        row[f"{metric}_mean"], row[f"{metric}_std"] = summarize(values)

    # The current logger writes these only as experiment-level summaries.
    pool_matches = [
        re.search(r"Average pooling time per run:\s+([0-9.eE+-]+)s", line)
        for line in lines
    ]
    memory_matches = [
        re.search(r"Average peak GPU memory reserved:\s+([0-9.eE+-]+) MB", line)
        for line in lines
    ]
    row["pooling_time_mean"] = next(
        (match.group(1) for match in reversed(pool_matches) if match), ""
    )
    row["gpu_memory_mean_mb"] = next(
        (match.group(1) for match in reversed(memory_matches) if match), ""
    )
    raw_rows = []
    for (seed, fold), run in sorted(runs.items()):
        raw_row = {
            "log_file": row["log_file"],
            "log_timestamp": row["log_timestamp"],
            "model": row["model"],
            "dataset": row["dataset"],
            "exp_name": row["exp_name"],
            "task": row["task"],
            "seed": seed,
            "fold": fold,
        }
        for metric in metric_names:
            raw_row[metric] = run.get(metric, "")
        raw_rows.append(raw_row)

    return row, raw_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        help="Experiment name, or a direct .log file/directory path",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path; defaults to results/<experiment>-log-stats.csv",
    )
    parser.add_argument(
        "--logs-root",
        default="logs",
        help="Root directory containing experiment log directories",
    )
    args = parser.parse_args()

    requested_experiment = None
    source = Path(args.experiment)
    if not source.exists():
        requested_experiment = args.experiment
        source = Path(args.logs_root) / requested_experiment

    if source.is_file():
        log_files = [source]
    elif source.is_dir():
        log_files = sorted(source.rglob("*.log"))
    else:
        raise SystemExit(f"Log path does not exist: {source}")

    rows = []
    for path in log_files:
        if path.stat().st_size == 0:
            continue
        row, raw_rows = parse_log(path)
        if requested_experiment and row["exp_name"] not in ("", requested_experiment):
            continue
        rows.append({"summary": row, "raw": raw_rows})
    if not rows:
        raise SystemExit("No usable log files found")

    # An experiment directory can contain reruns with the same method and
    # dataset. Keep the latest timestamped log for one final row per pair.
    if requested_experiment:
        latest = {}
        for item in rows:
            row = item["summary"]
            key = (row["model"], row["dataset"])
            previous = latest.get(key)
            if previous is None or (row["log_timestamp"], row["log_file"]) > (
                previous["summary"]["log_timestamp"], previous["summary"]["log_file"]
            ):
                latest[key] = item
        rows = sorted(
            latest.values(),
            key=lambda item: (item["summary"]["model"], item["summary"]["dataset"]),
        )

    summary_rows = [item["summary"] for item in rows]
    raw_rows = [raw for item in rows for raw in item["raw"]]
    summary_fieldnames = list(summary_rows[0].keys())
    raw_fieldnames = list(raw_rows[0].keys()) if raw_rows else []
    output = Path(args.output) if args.output else None
    if output is None and requested_experiment:
        output = Path("results") / f"{requested_experiment}-log-stats.csv"

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        if requested_experiment:
            raw_output = output.with_name(f"{requested_experiment}-raw-results.csv")
        else:
            raw_output = output.with_name(f"{output.stem}-raw-results.csv")
        with raw_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=raw_fieldnames)
            writer.writeheader()
            writer.writerows(raw_rows)
        print(f"Wrote {len(summary_rows)} method-dataset summaries to {output}")
        print(f"Wrote {len(raw_rows)} seed-fold results to {raw_output}")
    else:
        writer = csv.DictWriter(__import__("sys").stdout, fieldnames=summary_fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
