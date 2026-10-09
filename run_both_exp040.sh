#!/usr/bin/env bash
set -euo pipefail
CUDA_ECHO="${1:-0}"
CUDA_PRIVATE="${2:-0}"
bash run_exp040.sh echonext "${CUDA_ECHO}" EXP040_LateStartEMA_GlobalEnsemble_echonext
bash run_exp040.sh private_data "${CUDA_PRIVATE}" EXP040_LateStartEMA_GlobalEnsemble_private
