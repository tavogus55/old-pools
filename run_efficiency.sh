#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}" || exit 1

# Duan Pool classification methods. count1 is the sparse CountSketch variant.
# Sparse methods are listed first, followed by dense methods.
SPARSE_MODELS=(topk sag ndp graclus count1 count2 count4 ndrp)
DENSE_MODELS=(diff mincut unif gaus)
DATASETS=(DD IMDB-BINARY IMDB-MULTI MUTAG NCI1 NCI109 PROTEINS)
MODELS=("${SPARSE_MODELS[@]}" "${DENSE_MODELS[@]}")
EXP_NAME="${1:-efficiency}"

TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#DATASETS[@]}))

echo "========================================="
echo "Planned Duan Pool efficiency experiments"
echo "========================================="
echo "Sparse methods: ${SPARSE_MODELS[*]}"
echo "Dense methods : ${DENSE_MODELS[*]}"
echo "Datasets      : ${DATASETS[*]}"
echo "Experiment name: ${EXP_NAME}"
echo "Total         : ${TOTAL_EXPERIMENTS}"
echo ""
echo "Experiment order"
echo "----------------"

PLAN_INDEX=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        PLAN_INDEX=$((PLAN_INDEX + 1))
        echo "${PLAN_INDEX}: ${MODEL} ${DATASET}"
    done
done

echo ""
echo "========================================="
echo "Starting experiments"
echo "========================================="

CURRENT_EXPERIMENT=0
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        CURRENT_EXPERIMENT=$((CURRENT_EXPERIMENT + 1))

        echo ""
        echo "========================================="
        echo "Running experiment ${CURRENT_EXPERIMENT}/${TOTAL_EXPERIMENTS}"
        echo "Method  : ${MODEL}"
        echo "Dataset : ${DATASET}"
        echo "========================================="

        # No hyperparameters are supplied; main.py uses its defaults.
        python main.py \
            --methods "${MODEL}" \
            --dataset "${DATASET}" \
            --exp-name "${EXP_NAME}"

        EXIT_CODE=$?
        if [ "${EXIT_CODE}" -ne 0 ]; then
            echo "ERROR: ${MODEL} on ${DATASET} failed with exit code ${EXIT_CODE}"
            echo "Continuing to the next experiment..."
        fi
    done
done

echo ""
echo "========================================="
echo "All ${TOTAL_EXPERIMENTS} experiments completed"
echo "========================================="
