#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/codes/VGG_BatchNorm"
OUTPUT_ROOT="${1:-outputs_full}"
TRAIN_ITEMS="${TRAIN_ITEMS:--1}"
VAL_ITEMS="${VAL_ITEMS:--1}"
TEST_ITEMS="${TEST_ITEMS:--1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCH_SCALE="${EPOCH_SCALE:-1.0}"
SEED="${SEED:-2026}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_project2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESUME="${RESUME:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

echo "[1/5] project root: ${ROOT_DIR}"
echo "[2/5] activating environment: ${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[3/5] entering project directory"
cd "${PROJECT_DIR}"

EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-completed)
fi

echo "[4/5] running project pipeline"
"${PYTHON_BIN}" run_project2.py \
  --output-root "${OUTPUT_ROOT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --train-items "${TRAIN_ITEMS}" \
  --val-items "${VAL_ITEMS}" \
  --test-items "${TEST_ITEMS}" \
  --epoch-scale "${EPOCH_SCALE}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"

echo "[5/5] packaging results"
tar -czf "${OUTPUT_ROOT}.tar.gz" "${OUTPUT_ROOT}"

echo "Done."
echo "Results directory: ${PROJECT_DIR}/${OUTPUT_ROOT}"
echo "Packed archive: ${PROJECT_DIR}/${OUTPUT_ROOT}.tar.gz"
