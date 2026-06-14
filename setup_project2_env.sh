#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${ROOT_DIR}/codes/VGG_BatchNorm"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv_project2}"

echo "[1/4] project root: ${ROOT_DIR}"
echo "[2/4] creating virtual environment: ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "[3/4] upgrading pip"
python -m pip install --upgrade pip

echo "[4/4] installing dependencies"
python -m pip install -r "${PROJECT_DIR}/requirements.txt"

echo "Environment setup complete."
echo "Virtual environment: ${VENV_DIR}"
