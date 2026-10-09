#!/usr/bin/env bash
set -euo pipefail
DATASET="${1:-private_data}"
CUDA_ID="${2:-0}"
LOG_NAME="${3:-EXP040_LateStartEMA_GlobalEnsemble_${DATASET}}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ecg_fl/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then PYTHON_BIN="python"; fi
"${PYTHON_BIN}" main.py --dataset_name "${DATASET}" --cuda "${CUDA_ID}" --log_name "${LOG_NAME}"
