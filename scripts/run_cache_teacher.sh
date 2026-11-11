#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Cache the greedy teacher answer a_T for every training prompt (Section 3.2 of the proposal):
# "The teacher answers each prompt once (greedy) before training; its answer a_T is cached and never checked."
# Uses the chat template with Qwen3 non-thinking mode, vLLM if installed.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DATA_PATH=${DATA_PATH:-"${TRAIN_FILE}"}
OUTPUT_PATH=${OUTPUT_PATH:-"${TEACHER_CACHE}"}
MAX_TOKENS=${MAX_TOKENS:-2048}
CACHE_TP=${CACHE_TP:-1}

if [ ! -f "${DATA_PATH}" ]; then
  echo "Training data not found at ${DATA_PATH}. Preprocessing MATH..."
  mkdir -p "$(dirname "${DATA_PATH}")"
  python3 examples/data_preprocess/math_dataset.py --local_save_dir "$(dirname "${DATA_PATH}")"
fi

python3 scripts/cache_teacher_answers.py \
  --data_path "${DATA_PATH}" --model_path "${TEACHER_MODEL}" --output_path "${OUTPUT_PATH}" \
  --max_tokens "${MAX_TOKENS}" --tp "${CACHE_TP}" --backend "${CACHE_BACKEND:-auto}"
echo "Teacher answer cache: ${OUTPUT_PATH}"
