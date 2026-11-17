#!/usr/bin/env bash

set -euo pipefail

if [[ $# -eq 0 ]]; then
  source "$(dirname -- "${BASH_SOURCE[0]}")/load_runner_config.sh" experiments count
  for ((experiment_index = 0; experiment_index < EXPERIMENT_COUNT; experiment_index++)); do
    echo "Starting experiment $((experiment_index + 1))/${EXPERIMENT_COUNT}"
    bash "${BASH_SOURCE[0]}" "${experiment_index}"
  done
  echo "All ${EXPERIMENT_COUNT} queued experiments completed."
  exit 0
fi

source "$(dirname -- "${BASH_SOURCE[0]}")/load_runner_config.sh" experiments "$1"

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

datasets=()
models=()
IFS=',' read -r -a datasets <<< "${DATASETS}"
IFS=',' read -r -a models <<< "${MODELS}"

if [[ ${#datasets[@]} -eq 0 || ${#models[@]} -eq 0 ]]; then
  echo "Experiments require nonempty datasets and models in the config." >&2
  exit 1
fi

training_args=()
for option in \
  EPOCHS:--epochs FOLDS:--k-folds SEEDS:--seeds EARLY_STOP:--early_stop \
  MP_LAYER:--mp-layer LEARNING_RATE:--lr WEIGHT_DECAY:--weight_decay \
  HIDDEN:--hidden POOLING_RATIO:--pratio DROPOUT:--dropout BATCH_SIZE:--batch-size; do
  variable="${option%%:*}"
  flag="${option#*:}"
  if [[ -n "${!variable:-}" ]]; then
    training_args+=("${flag}" "${!variable}")
  fi
done
if [[ "${USE_CUDA:-0}" == "1" ]]; then
  training_args+=(--cuda)
fi

pair_count=$(( ${#datasets[@]} * ${#models[@]} ))
echo "Experiment: ${EXP_NAME}"
echo "Mode: default hyperparameters from main.py (unless explicitly overridden)"
echo "Dataset-model pairs: ${pair_count}"

combination=0
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    combination=$((combination + 1))
    echo "Running ${combination}/${pair_count} - ${model} on ${dataset}..."
    "${PYTHON_BIN}" main.py \
      --dataset "${dataset}" \
      --model "${model}" \
      --exp_name "${EXP_NAME}" \
      "${training_args[@]}"
  done
done

echo "All ${pair_count} model/dataset experiments completed."
