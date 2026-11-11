#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Phase 0 pilot without training (Section "Phase 0" of the proposal): decides whether the project continues.
# Needs the MATH train parquet and the a_T cache. Runs on one GPU with >= 80GB (student+teacher in vLLM and HF);
# use CUDA_VISIBLE_DEVICES to pick it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

NUM_PROMPTS=${NUM_PROMPTS:-100}
M_SCREEN=${M_SCREEN:-8}
M_ADAPTIVE=${M_ADAPTIVE:-32}
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/phase0_pilot"}

require_file "${TRAIN_FILE}" "Run: bash scripts/run_all_experiments_master.sh data"
ensure_teacher_cache
mkdir -p "${OUTPUT_DIR}"

python3 phase0/run_phase0_pilot.py \
  --student_model "${STUDENT_MODEL}" --teacher_model "${TEACHER_MODEL}" \
  --data_path "${TRAIN_FILE}" --teacher_cache "${TEACHER_CACHE}" \
  --num_prompts "${NUM_PROMPTS}" --m_screen "${M_SCREEN}" --m_adaptive "${M_ADAPTIVE}" \
  --seed "${SEED}" --output_dir "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/phase0.log"

echo "Report: ${OUTPUT_DIR}/phase0_report.md   Results: ${OUTPUT_DIR}/phase0_results.json"
