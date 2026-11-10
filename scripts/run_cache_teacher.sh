#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Script to cache greedy teacher answers a_T before training.
# Section 3.2: "The teacher answers each prompt once (greedy) before training;
# its answer a_T is cached and never checked."

set -euo pipefail

# Environment / Paths
DATA_PATH=${DATA_PATH:-"data/math/train.parquet"}
TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen3-8B"}
OUTPUT_PATH=${OUTPUT_PATH:-"data/math/teacher_answers_qwen3_8b.json"}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_TOKENS=${MAX_TOKENS:-2048}
DEVICE=${DEVICE:-"cuda"}

echo "=========================================================="
echo "Caching Teacher Answers a_T"
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Data Path:     ${DATA_PATH}"
echo "Output Cache:  ${OUTPUT_PATH}"
echo "=========================================================="

# Check if data exists; if not, run preprocessing
if [ ! -f "${DATA_PATH}" ]; then
    echo "Training data not found at ${DATA_PATH}. Preprocessing MATH dataset..."
    mkdir -p data/math
    python3 examples/data_preprocess/math_dataset.py \
        --local_save_dir data/math
fi

# Run caching
python3 scripts/cache_teacher_answers.py \
    --data_path "${DATA_PATH}" \
    --model_path "${TEACHER_MODEL}" \
    --output_path "${OUTPUT_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_tokens "${MAX_TOKENS}" \
    --device "${DEVICE}"

echo "Teacher answer caching complete!"
