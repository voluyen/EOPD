#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Evaluation script across MATH500, AIME24, AIME25, AMC23, Minerva Math, OlympiadBench.
# Protocol: Qwen2.5-Math (temp 1.0, top-p 0.8, 8192 tokens, 8 samples), Avg@8 & Pass@8.

set -euo pipefail

MODEL_PATH=${1:-${MODEL_PATH:-"checkpoints/t1_ours_cw_opd/global_step_final"}}
BENCHMARKS_DIR=${BENCHMARKS_DIR:-"data/eval_benchmarks"}
OUTPUT_PATH=${OUTPUT_PATH:-"${MODEL_PATH}/eval_results.json"}
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.8}
MAX_TOKENS=${MAX_TOKENS:-8192}
DEVICE=${DEVICE:-"cuda"}

echo "=========================================================="
echo "Evaluating Model on Reasoning Benchmarks"
echo "Model Path:     ${MODEL_PATH}"
echo "Benchmarks Dir: ${BENCHMARKS_DIR}"
echo "Output Path:    ${OUTPUT_PATH}"
echo "Samples:        ${N_SAMPLES}"
echo "=========================================================="

# Ensure benchmark datasets exist
if [ ! -d "${BENCHMARKS_DIR}" ] || [ -z "$(ls -A ${BENCHMARKS_DIR}/*.parquet 2>/dev/null)" ]; then
    echo "Benchmark datasets not found in ${BENCHMARKS_DIR}. Preparing datasets..."
    python3 scripts/prepare_eval_benchmarks.py --output_dir "${BENCHMARKS_DIR}"
fi

python3 eval/evaluate_benchmarks.py \
    --model_path "${MODEL_PATH}" \
    --benchmarks_dir "${BENCHMARKS_DIR}" \
    --output_path "${OUTPUT_PATH}" \
    --n_samples "${N_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}" \
    --device "${DEVICE}"

echo "Benchmark evaluation completed! Results saved to ${OUTPUT_PATH}"
