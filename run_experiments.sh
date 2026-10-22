#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-rw-env}"
CONDA_EXE="${CONDA_EXE:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Find a Python interpreter with PyTorch, activating Conda when necessary.
python_has_torch() {
  local candidate="$1"
  command -v "${candidate}" >/dev/null 2>&1 || return 1
  "${candidate}" -c "import torch" >/dev/null 2>&1
}

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3 python python.exe; do
    if python_has_torch "${candidate}"; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  conda_script=""

  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [[ -f "${conda_base}/etc/profile.d/conda.sh" ]]; then
      conda_script="${conda_base}/etc/profile.d/conda.sh"
    fi
  fi

  if [[ -z "${conda_script}" ]]; then
    for conda_root in \
      "${HOME}/miniconda3" \
      "${HOME}/anaconda3" \
      "/mnt/Miniconda3" \
      "/c/Users/${USERNAME:-${USER:-user}}/miniconda3" \
      "/c/Users/${USERNAME:-${USER:-user}}/anaconda3"; do
      if [[ -f "${conda_root}/etc/profile.d/conda.sh" ]]; then
        conda_script="${conda_root}/etc/profile.d/conda.sh"
        break
      fi
    done
  fi

  if [[ -z "${CONDA_EXE}" ]]; then
    for conda_executable in \
      "/mnt/Miniconda3/bin/conda" \
      "${HOME}/miniconda3/bin/conda" \
      "${HOME}/anaconda3/bin/conda"; do
      if [[ -x "${conda_executable}" ]]; then
        CONDA_EXE="${conda_executable}"
        break
      fi
    done
  fi

  if [[ -n "${conda_script}" ]]; then
    # shellcheck disable=SC1090
    source "${conda_script}"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -n "${CONDA_EXE}" && -x "${CONDA_EXE}" ]]; then
    eval "$("${CONDA_EXE}" shell.bash hook)"
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "${CONDA_ENV}"
    if python_has_torch python; then
      PYTHON_BIN="python"
    fi
  fi
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Could not find a Python interpreter with PyTorch." >&2
  echo "Set PYTHON_BIN explicitly or install/activate the '${CONDA_ENV}' Conda environment." >&2
  exit 1
fi

# Set USE_TUNED=1 to load every dataset-model row from one tuning manifest.
USE_TUNED="${USE_TUNED:-0}"
TUNING_EXPERIMENT="${TUNING_EXPERIMENT:-tuning}"
TUNING_MANIFEST="${TUNING_MANIFEST:-results/hyperparameter_tuning/${TUNING_EXPERIMENT}/${TUNING_EXPERIMENT}_tuning_manifest.csv}"

folds="${FOLDS:-10}"
seeds="${SEEDS:-42}"
exp_name="${EXP_NAME:-experiment}"

# Manual mode configuration. These values are used when USE_TUNED=0.
datasets=("MUTAG")
models=("ndrp")

# Optional comma-separated command-line overrides, for example:
# DATASETS=DD,PROTEINS MODELS=ndrp,topk bash run_experiments.sh
if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a datasets <<< "${DATASETS}"
fi
if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -r -a models <<< "${MODELS}"
fi
learning_rate="0.005"
weight_decay="0.001"
hidden=32
pooling_ratio="0.5"
dropout="0.0"
batch_size=128

# Experiment rows are stored as tab-separated values:
# dataset, model, learning rate, weight decay, dropout, batch size, hidden, ratio.
experiment_rows=()

if [[ "${USE_TUNED}" == "1" ]]; then
  [[ -f "${TUNING_MANIFEST}" ]] || {
    echo "Tuning manifest not found: ${TUNING_MANIFEST}" >&2
    echo "Set TUNING_EXPERIMENT or TUNING_MANIFEST to the correct manifest." >&2
    exit 1
  }

  # Use Python's CSV reader so quoted PowerShell manifests are handled safely.
  mapfile -t experiment_rows < <(
    "${PYTHON_BIN}" - "${TUNING_MANIFEST}" <<'PY'
import csv
import os
import sys

manifest_path = sys.argv[1]
required = [
    "dataset",
    "model",
    "selected_learning_rate",
    "selected_weight_decay",
    "selected_dropout",
    "selected_batch_size",
    "hidden",
    "pratio",
]

with open(manifest_path, newline="") as handle:
    rows = csv.DictReader(handle)
    missing = [field for field in required if field not in (rows.fieldnames or [])]
    if missing:
        raise SystemExit("Manifest is missing columns: " + ", ".join(missing))
    dataset_filter = {
        value for value in os.environ.get("DATASETS", "").split(",") if value
    }
    model_filter = {
        value for value in os.environ.get("MODELS", "").split(",") if value
    }
    for row in rows:
        if dataset_filter and row["dataset"] not in dataset_filter:
            continue
        if model_filter and row["model"] not in model_filter:
            continue
        print("\t".join(row[field] for field in required))
PY
  )

  [[ ${#experiment_rows[@]} -gt 0 ]] || {
    echo "The tuning manifest contains no experiment rows: ${TUNING_MANIFEST}" >&2
    exit 1
  }
else
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      experiment_rows+=(
        "${dataset}"$'\t'"${model}"$'\t'"${learning_rate}"$'\t'"${weight_decay}"$'\t'"${dropout}"$'\t'"${batch_size}"$'\t'"${hidden}"$'\t'"${pooling_ratio}"
      )
    done
  done
fi

pair_count=${#experiment_rows[@]}
seed_count=$(tr ',' ' ' <<< "${seeds}" | wc -w | tr -d ' ')
total_seed_fold_runs=$((pair_count * folds * seed_count))

echo "Experiment plan"
echo "---------------"
if [[ "${USE_TUNED}" == "1" ]]; then
  echo "Mode: tuned manifest"
  echo "Manifest: ${TUNING_MANIFEST}"
else
  echo "Mode: manual configuration"
fi
echo "Dataset-model pairs: ${pair_count}"
echo "Folds per pair: ${folds}"
echo "Seeds: ${seeds}"
echo "Total seed-fold runs: ${total_seed_fold_runs}"
echo ""
echo "Pairs:"
for row in "${experiment_rows[@]}"; do
  IFS=$'\t' read -r dataset model lr wd do_value bs hidden_value ratio <<< "${row}"
  echo "- ${dataset} + ${model} (lr=${lr}, wd=${wd}, dropout=${do_value}, batch=${bs}, hidden=${hidden_value}, pratio=${ratio})"
done
echo ""

combination=0
for row in "${experiment_rows[@]}"; do
  IFS=$'\t' read -r dataset model lr wd do_value bs hidden_value ratio <<< "${row}"
  combination=$((combination + 1))

  echo "Running ${combination}/${pair_count} - ${model} on ${dataset} (${folds} folds x ${seed_count} seeds)..."
  "${PYTHON_BIN}" main.py \
    --dataset "${dataset}" \
    --model "${model}" \
    --k-folds "${folds}" \
    --seeds "${seeds}" \
    --lr "${lr}" \
    --weight_decay "${wd}" \
    --hidden "${hidden_value}" \
    --pratio "${ratio}" \
    --dropout "${do_value}" \
    --batch-size "${bs}" \
    --exp_name "${exp_name}"
done

echo "All ${pair_count} model/dataset experiments completed."
