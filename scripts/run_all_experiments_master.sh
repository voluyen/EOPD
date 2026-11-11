#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Master Runner Script for Consequence-Weighted OPD Experiments.
# Proposal: "Where Does the Student Lose the Teacher's Conclusion?
# Teacher Answer-Belief Shifts as a Verifier-Free Allocation Signal for On-Policy Distillation"

set -euo pipefail

# Default environment configuration
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/common.sh          # STUDENT_MODEL, TEACHER_MODEL, N_GPUS, DATA_DIR, TEACHER_CACHE, ...
export STUDENT_MODEL TEACHER_MODEL N_GPUS NNODES DATA_DIR TEACHER_CACHE
export BENCHMARKS_DIR=${BENCHMARKS_DIR:-"data/eval_benchmarks"}
SEEDS=${SEEDS:-"0 1 2"}          # Section 6: 3 seeds

print_banner() {
    gpu_count=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU" || true)
    if [ "${gpu_count}" -eq 0 ] 2>/dev/null; then
        gpu_count="0 (CPU/No Driver)"
    fi
    echo "======================================================================"
    echo "  CONSEQUENCE-WEIGHTED ON-POLICY DISTILLATION (CW-OPD) MASTER SUITE"
    echo "======================================================================"
    echo "  Student Model:  ${STUDENT_MODEL}"
    echo "  Teacher Model:  ${TEACHER_MODEL}"
    echo "  GPUs Detected:  ${gpu_count}"
    echo "  Target GPUs:    ${N_GPUS} x ${NNODES} node(s)"
    echo "  Teacher Cache:  ${TEACHER_CACHE}"
    echo "======================================================================"
}

cmd_env_check() {
    echo ""
    echo ">>> [Step 0] Environment & Dependency Check..."
    python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device Count: {torch.cuda.device_count()}, Name: {torch.cuda.get_device_name(0)}')
try:
    import verl
    print('verl: OK')
except Exception as e:
    print(f'verl import: {e}')
try:
    import math_verify
    print('math_verify: OK')
except Exception as e:
    print(f'math_verify: {e}')
"
    echo "Unit test verification..."
    pytest tests/test_cw_opd.py -q
    echo "Environment check PASSED!"
}

cmd_prepare_data() {
    echo ""
    echo ">>> [Step 1] Preparing Datasets..."
    mkdir -p "${DATA_DIR}" "${BENCHMARKS_DIR}"
    
    if [ ! -f "${DATA_DIR}/train.parquet" ]; then
        echo "Preparing MATH dataset..."
        python3 examples/data_preprocess/math_dataset.py --local_save_dir "${DATA_DIR}"
    else
        echo "MATH dataset already exists at ${DATA_DIR}/train.parquet."
    fi

    echo "Preparing Evaluation Benchmarks..."
    python3 scripts/prepare_eval_benchmarks.py --output_dir "${BENCHMARKS_DIR}"
    echo "Data preparation complete!"
}

cmd_setup() {
    echo ""; echo ">>> [Step -2] Server setup"; bash scripts/setup_server.sh
}

cmd_models() {
    echo ""; echo ">>> [Step -1] Downloading models"; bash scripts/download_models.sh
}

cmd_cache_teacher() {
    echo ""
    echo ">>> [Step 2] Caching Teacher Answers a_T..."
    bash scripts/run_cache_teacher.sh
}

cmd_phase0() {
    echo ""
    echo ">>> [Step 3] Running Phase 0 Pilot (Pilot Without Training)..."
    bash scripts/run_phase0_pilot.sh
    echo "Read experiments/phase0_pilot/phase0_report.md and decide GO/NO-GO before running table1/table2."
}

run_for_seeds() {   # <script> : run a training script for every seed
    for seed in ${SEEDS}; do
        echo "--- $1  seed=${seed}"
        SEED="${seed}" bash "$1"
    done
}

cmd_table1() {
    echo ""
    echo ">>> [Step 4] Table 1 (OPD, EOPD, FUW, ours, ours+EOPD, permutation control) x seeds: ${SEEDS}"
    for s in run_train_t1_opd run_train_t1_eopd run_train_t1_fuw run_train_t1_ours_cw_opd \
             run_train_t1_ours_cw_eopd run_train_t1_permutation; do
        run_for_seeds "scripts/${s}.sh"
    done
}

cmd_table2() {
    echo ""
    echo ">>> [Step 5] Table 2 ablations x seeds: ${SEEDS}"
    for s in run_train_t2_ablation_at_only run_train_t2_ablation_length_norm run_train_t2_ablation_kl; do
        run_for_seeds "scripts/${s}.sh"
    done
}

cmd_eval() {
    echo ""
    echo ">>> [Step 6] Benchmark evaluation (untrained student + every finished run)"
    if [ ! -f "${OUTPUT_ROOT}/student_base_eval.json" ]; then
        OUTPUT_PATH="${OUTPUT_ROOT}/student_base_eval.json" bash scripts/run_eval_all_benchmarks.sh "${STUDENT_MODEL}" \
            || echo "Warning: eval failed for the base student"
    fi
    for run_dir in "${OUTPUT_ROOT}"/t[12]_*_seed*; do
        [ -f "${run_dir}/latest_checkpointed_iteration.txt" ] || continue
        [ -f "${run_dir}/eval_results.json" ] && { echo "skip ${run_dir} (already evaluated)"; continue; }
        bash scripts/run_eval_all_benchmarks.sh "${run_dir}" || echo "Warning: eval failed on ${run_dir}"
    done
    echo "Paired bootstrap, e.g.:"
    echo "  python eval/compare_results.py --baseline ${OUTPUT_ROOT}/t1_eopd_seed{0,1,2}/eval_results.json \\"
    echo "         --method ${OUTPUT_ROOT}/t1_ours_cw_eopd_seed{0,1,2}/eval_results.json"
}

print_banner

STEP=${1:-"all"}

case "${STEP}" in
    setup)
        cmd_setup
        ;;
    models)
        cmd_models
        ;;
    check)
        cmd_env_check
        ;;
    data)
        cmd_prepare_data
        ;;
    cache)
        cmd_cache_teacher
        ;;
    phase0)
        cmd_phase0
        ;;
    table1)
        cmd_table1
        ;;
    table2)
        cmd_table2
        ;;
    eval)
        cmd_eval
        ;;
    all)
        echo "Running complete end-to-end pipeline..."
        cmd_models
        cmd_env_check
        cmd_prepare_data
        cmd_cache_teacher
        cmd_phase0
        cmd_table1
        cmd_table2
        cmd_eval
        printf "\n======================================================================\n"
        echo "  ALL EXPERIMENTS FINISHED SUCCESSFULLY!"
        echo "======================================================================"
        ;;
    *)
        echo "Usage: bash scripts/run_all_experiments_master.sh [setup|models|check|data|cache|phase0|table1|table2|eval|all]"
        exit 1
        ;;
esac
