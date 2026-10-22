#!/usr/bin/env bash

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-rw-env}"
CONDA_EXE="${CONDA_EXE:-}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Find a Python interpreter with PyTorch. This lets the script run directly
# from Git Bash or Linux without requiring manual Conda activation first.
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
    # Some Linux installations do not expose conda.sh, but support the
    # shell-hook initialization required before `conda activate`.
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
EPOCHS="${EPOCHS:-50}"
FOLDS="${FOLDS:-10}"
SEEDS="${SEEDS:-42}"
EARLY_STOP="${EARLY_STOP:-10}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-tuning}"
MP_LAYER="${MP_LAYER:-graphconv}"
HIDDEN="${HIDDEN:-32}"
POOLING_RATIO="${POOLING_RATIO:-0.5}"

datasets=(
  "AIDS"
  "BZR"
  "COX2"
  "DD"
  "DHFR"
  "ENZYMES"
  "IMDB-BINARY"
  "MSRC_21"
  "MSRC_9"
  "MUTAG"
  "PTC_MR"
  "Synthie"
)
# Node-clustering methods included in the tuning workflow.
models=("gaus" "unif" "count1" "count2" "count4" "ndrp")

# Optional comma-separated command-line overrides, for example:
# DATASETS=DD MODELS=gaus,unif bash run_hyperparameter_tuning.sh
if [[ -n "${DATASETS:-}" ]]; then
  IFS=',' read -r -a datasets <<< "${DATASETS}"
fi
if [[ -n "${MODELS:-}" ]]; then
  IFS=',' read -r -a models <<< "${MODELS}"
fi

# Coordinate-search candidates: 6 + 4 + 3 + 4 = 17 per model-dataset pair.
learning_rates=(0.01 0.05 0.001 0.005 0.0001 0.0005)
weight_decays=(0.01 0.001 0.0001 0.00001)
dropouts=(0.0 0.2 0.5)
batch_sizes=(32 64 128 256)

# Set QUICK_TEST=1 to run one candidate per model-dataset pair while checking
# output paths. The normal coordinate-search arrays above remain unchanged.
QUICK_TEST="${QUICK_TEST:-0}"
if [[ "${QUICK_TEST}" == "1" ]]; then
  # The shortest path test uses only the first method and one configuration.
  if [[ -z "${MODELS:-}" ]]; then
    models=("gaus")
  fi
  learning_rates=(0.001)
  weight_decays=(0.0001)
  dropouts=(0.2)
  batch_sizes=(128)
fi

# Set USE_CUDA=1 to append --cuda to every run.
USE_CUDA="${USE_CUDA:-0}"

tuning_root="results/hyperparameter_tuning"
experiment_directory="${tuning_root}/${EXPERIMENT_NAME}"
candidate_directory="${experiment_directory}/candidates"
final_directory="${experiment_directory}/final"
mkdir -p "${candidate_directory}" "${final_directory}"
manifest_file="${experiment_directory}/${EXPERIMENT_NAME}_tuning_manifest.csv"
if [[ ! -f "${manifest_file}" ]]; then
  echo "dataset,model,selected_learning_rate,selected_weight_decay,selected_dropout,selected_batch_size,hidden,pratio,folds,folds_used,seeds,epochs,early_stop,final_experiment,final_result" > "${manifest_file}"
fi

if [[ "${QUICK_TEST}" == "1" ]]; then
  candidate_count=1
else
  candidate_count=$(( ${#learning_rates[@]} + ${#weight_decays[@]} + ${#dropouts[@]} + ${#batch_sizes[@]} ))
fi
total_runs=$(( ${#datasets[@]} * ${#models[@]} * candidate_count ))
run_number=0

csv_field() {
  local file="$1"
  local field="$2"
  awk -F, -v requested_field="${field}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == requested_field) column = i
      }
    }
    NR == 2 {
      print $column
      exit
    }
  ' "${file}"
}

manifest_has_pair() {
  local dataset="$1"
  local model="$2"
  awk -F, -v requested_dataset="${dataset}" -v requested_model="${model}" \
    'NR > 1 && $1 == requested_dataset && $2 == requested_model { found = 1 } END { exit(found ? 0 : 1) }' \
    "${manifest_file}"
}

is_regression_dataset() {
  case "${1^^}" in
    ESOL|FREESOLV|LIPO|BACE|QM7|QM7B|QM8|MOLHIV|MOLPCBA) return 0 ;;
    *) return 1 ;;
  esac
}

run_candidate() {
  local dataset="$1"
  local model="$2"
  local stage="$3"
  local lr="$4"
  local weight_decay="$5"
  local dropout="$6"
  local batch_size="$7"

  run_number=$((run_number + 1))
  local configuration="lr${lr}_wd${weight_decay}_do${dropout}_bs${batch_size}"
  local candidate_name="${EXPERIMENT_NAME}_candidate_${dataset}_${model}_${stage}_${configuration}"
  local result_file="results/${candidate_name}.csv"
  local raw_result_file="results/${candidate_name}_raw.csv"
  local stored_result_file="${candidate_directory}/${candidate_name}.csv"

  echo "Tuning ${run_number}/${total_runs} - ${model} on ${dataset} - ${stage} - ${configuration}"

  # Resume support: a completed candidate is identified by its stored CSV.
  # Reuse it instead of launching main.py again.
  if [[ ! -f "${stored_result_file}" && -f "${result_file}" ]]; then
    mv -f "${result_file}" "${stored_result_file}"
    if [[ -f "${raw_result_file}" ]]; then
      mv -f "${raw_result_file}" "${candidate_directory}/${candidate_name}_raw.csv"
    fi
  fi

  if [[ -f "${stored_result_file}" ]]; then
    echo "  Resuming: reusing ${stored_result_file}"
    if is_regression_dataset "${dataset}"; then
      CANDIDATE_SCORE="$(csv_field "${stored_result_file}" avg_best_val_mse)"
    else
      CANDIDATE_SCORE="$(csv_field "${stored_result_file}" avg_best_val_acc)"
    fi
    [[ -n "${CANDIDATE_SCORE}" ]] || {
      echo "Stored candidate has no validation score: ${stored_result_file}" >&2
      exit 1
    }
    CANDIDATE_NAME="${candidate_name}"
    return
  fi

  local args=(
    main.py
    --dataset "${dataset}"
    --model "${model}"
    --epochs "${EPOCHS}"
    --k-folds "${FOLDS}"
    --folds-to-run 1
    --seeds "${SEEDS}"
    --early_stop "${EARLY_STOP}"
    --exp_name "${candidate_name}"
    --mp-layer "${MP_LAYER}"
    --lr "${lr}"
    --weight_decay "${weight_decay}"
    --hidden "${HIDDEN}"
    --pratio "${POOLING_RATIO}"
    --dropout "${dropout}"
    --batch-size "${batch_size}"
  )
  if [[ "${USE_CUDA}" == "1" ]]; then
    args+=(--cuda)
  fi

  "${PYTHON_BIN}" "${args[@]}"

  [[ -f "${result_file}" ]] || {
    echo "Expected tuning result was not created: ${result_file}" >&2
    exit 1
  }

  mv -f "${result_file}" "${stored_result_file}"
  if [[ -f "${raw_result_file}" ]]; then
    mv -f "${raw_result_file}" "${candidate_directory}/${candidate_name}_raw.csv"
  fi

  if is_regression_dataset "${dataset}"; then
    CANDIDATE_SCORE="$(csv_field "${stored_result_file}" avg_best_val_mse)"
  else
    CANDIDATE_SCORE="$(csv_field "${stored_result_file}" avg_best_val_acc)"
  fi
  CANDIDATE_NAME="${candidate_name}"
}

select_stage() {
  local dataset="$1"
  local model="$2"
  local stage="$3"
  shift 3
  local best_score=""
  local best_value=""

  for value in "$@"; do
    local trial_lr="${selected_lr}"
    local trial_weight_decay="${selected_weight_decay}"
    local trial_dropout="${selected_dropout}"
    local trial_batch_size="${selected_batch_size}"

    case "${stage}" in
      lr) trial_lr="${value}" ;;
      weight_decay) trial_weight_decay="${value}" ;;
      dropout) trial_dropout="${value}" ;;
      batch_size) trial_batch_size="${value}" ;;
    esac

    run_candidate "${dataset}" "${model}" "${stage}" \
      "${trial_lr}" "${trial_weight_decay}" "${trial_dropout}" "${trial_batch_size}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "${CANDIDATE_NAME}" "${stage}" "${trial_lr}" "${trial_weight_decay}" \
      "${trial_dropout}" "${trial_batch_size}" "${CANDIDATE_SCORE}" \
      "${dataset}" "${model}" >> "${stage_summary_file}"

    if [[ -z "${best_score}" ]]; then
      best_score="${CANDIDATE_SCORE}"
      best_value="${value}"
    elif is_regression_dataset "${dataset}"; then
      if awk -v current="${CANDIDATE_SCORE}" -v best="${best_score}" 'BEGIN { exit !(current < best) }'; then
        best_score="${CANDIDATE_SCORE}"
        best_value="${value}"
      fi
    elif awk -v current="${CANDIDATE_SCORE}" -v best="${best_score}" 'BEGIN { exit !(current > best) }'; then
      best_score="${CANDIDATE_SCORE}"
      best_value="${value}"
    fi
  done

  case "${stage}" in
    lr) selected_lr="${best_value}" ;;
    weight_decay) selected_weight_decay="${best_value}" ;;
    dropout) selected_dropout="${best_value}" ;;
    batch_size) selected_batch_size="${best_value}" ;;
  esac
}

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    selected_lr="0.001"
    selected_weight_decay="0.0001"
    selected_dropout="0.2"
    selected_batch_size="128"

    stage_summary_file="${candidate_directory}/${EXPERIMENT_NAME}_candidates_${dataset}_${model}.csv"
    if [[ ! -f "${stage_summary_file}" ]]; then
      echo "candidate,stage,learning_rate,weight_decay,dropout,batch_size,validation_score,dataset,model" > "${stage_summary_file}"
    fi

    if [[ "${QUICK_TEST}" == "1" ]]; then
      # Run exactly one candidate before the final retraining, solely to
      # verify that files are moved into the tuning directory correctly.
      run_candidate "${dataset}" "${model}" "quick" \
        "${selected_lr}" "${selected_weight_decay}" \
        "${selected_dropout}" "${selected_batch_size}"
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${CANDIDATE_NAME}" "quick" "${selected_lr}" "${selected_weight_decay}" \
        "${selected_dropout}" "${selected_batch_size}" "${CANDIDATE_SCORE}" \
        "${dataset}" "${model}" >> "${stage_summary_file}"
    else
      select_stage "${dataset}" "${model}" lr "${learning_rates[@]}"
      select_stage "${dataset}" "${model}" weight_decay "${weight_decays[@]}"
      select_stage "${dataset}" "${model}" dropout "${dropouts[@]}"
      select_stage "${dataset}" "${model}" batch_size "${batch_sizes[@]}"
    fi

    final_name="${EXPERIMENT_NAME}_best_${dataset}_${model}"
    final_result="results/${final_name}.csv"
    final_raw_result="results/${final_name}_raw.csv"
    echo "Selected for ${model} on ${dataset}: lr=${selected_lr}, wd=${selected_weight_decay}, dropout=${selected_dropout}, batch_size=${selected_batch_size}"

    # If the final result and manifest row already exist, this pair is fully
    # complete and can be skipped during a resumed tuning run.
    if manifest_has_pair "${dataset}" "${model}" && [[ -f "${final_directory}/${final_name}.csv" ]]; then
      echo "  Resuming: final result already exists for ${model} on ${dataset}"
      continue
    fi

    final_args=(
      main.py
      --dataset "${dataset}"
      --model "${model}"
      --epochs "${EPOCHS}"
      --k-folds "${FOLDS}"
      --folds-to-run 1
      --seeds "${SEEDS}"
      --early_stop "${EARLY_STOP}"
      --exp_name "${final_name}"
      --mp-layer "${MP_LAYER}"
      --lr "${selected_lr}"
      --weight_decay "${selected_weight_decay}"
      --hidden "${HIDDEN}"
      --pratio "${POOLING_RATIO}"
      --dropout "${selected_dropout}"
      --batch-size "${selected_batch_size}"
      --retrain-on-train-val
    )
    if [[ "${USE_CUDA}" == "1" ]]; then
      final_args+=(--cuda)
    fi
    "${PYTHON_BIN}" "${final_args[@]}"

    [[ -f "${final_result}" ]] || {
      echo "Expected final result was not created: ${final_result}" >&2
      exit 1
    }
    mv -f "${final_result}" "${final_directory}/${final_name}.csv"
    if [[ -f "${final_raw_result}" ]]; then
      mv -f "${final_raw_result}" "${final_directory}/${final_name}_raw.csv"
    fi

    if ! manifest_has_pair "${dataset}" "${model}"; then
      printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${dataset}" "${model}" "${selected_lr}" "${selected_weight_decay}" \
        "${selected_dropout}" "${selected_batch_size}" "${HIDDEN}" "${POOLING_RATIO}" \
        "${FOLDS}" "1" "${SEEDS}" "${EPOCHS}" "${EARLY_STOP}" "${final_name}" \
        "${final_directory}/${final_name}.csv" >> "${manifest_file}"
    fi
  done
done

echo "Coordinate hyperparameter tuning completed: ${total_runs} candidate runs."
echo "Tuning artifacts are stored in ${experiment_directory}."
