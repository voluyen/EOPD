#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Script to run Phase 0 Pilot without training (Section 5 of Proposal).
# Decides whether the project continues based on incremental predictive value
# before the answer appears.

set -euo pipefail

STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3-1.7B-Base"}
TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen3-8B"}
NUM_PROMPTS=${NUM_PROMPTS:-100}
M_SCREEN=${M_SCREEN:-8}
M_ADAPTIVE=${M_ADAPTIVE:-32}
OUTPUT_DIR=${OUTPUT_DIR:-"experiments/phase0_pilot"}

echo "=========================================================="
echo "Running Phase 0 Pilot (Pilot Without Training)"
echo "Student Model: ${STUDENT_MODEL}"
echo "Teacher Model: ${TEACHER_MODEL}"
echo "Num Prompts:   ${NUM_PROMPTS}"
echo "M Screen:      ${M_SCREEN}"
echo "M Adaptive:    ${M_ADAPTIVE}"
echo "Output Dir:    ${OUTPUT_DIR}"
echo "=========================================================="

mkdir -p "${OUTPUT_DIR}"

python3 phase0/run_phase0_pilot.py \
    --student_model "${STUDENT_MODEL}" \
    --teacher_model "${TEACHER_MODEL}" \
    --num_prompts "${NUM_PROMPTS}" \
    --m_screen "${M_SCREEN}" \
    --m_adaptive "${M_ADAPTIVE}" \
    --output_dir "${OUTPUT_DIR}"

echo "=========================================================="
echo "Phase 0 Pilot Completed!"
echo "Report: ${OUTPUT_DIR}/phase0_report.md"
echo "Results JSON: ${OUTPUT_DIR}/phase0_results.json"
echo "=========================================================="
