#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Master Runner Script for Consequence-Weighted OPD Experiments.
# Proposal: "Where Does the Student Lose the Teacher's Conclusion?
# Teacher Answer-Belief Shifts as a Verifier-Free Allocation Signal for On-Policy Distillation"

set -euo pipefail

# Default environment configuration
export STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3-1.7B-Base"}
export TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen3-8B"}
export N_GPUS=${N_GPUS:-4}
export NNODES=${NNODES:-1}
export DATA_DIR=${DATA_DIR:-"data/math"}
export BENCHMARKS_DIR=${BENCHMARKS_DIR:-"data/eval_benchmarks"}
export TEACHER_CACHE=${TEACHER_CACHE:-"data/math/teacher_answers_qwen3_8b.json"}

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

cmd_cache_teacher() {
    echo ""
    echo ">>> [Step 2] Caching Teacher Answers a_T..."
    bash scripts/run_cache_teacher.sh
}

cmd_phase0() {
    echo ""
    echo ">>> [Step 3] Running Phase 0 Pilot (Pilot Without Training)..."
    bash scripts/run_phase0_pilot.sh
}

cmd_table1() {
    echo ""
    echo ">>> [Step 4] Running Table 1 Main Comparison Experiments..."
    echo "1/6: Running OPD baseline..."
    bash scripts/run_train_t1_opd.sh

    echo "2/6: Running EOPD baseline..."
    bash scripts/run_train_t1_eopd.sh

    echo "3/6: Running FUW baseline..."
    bash scripts/run_train_t1_fuw.sh

    echo "4/6: Running Ours (CW-OPD)..."
    bash scripts/run_train_t1_ours_cw_opd.sh

    echo "5/6: Running Ours + EOPD (CW-EOPD)..."
    bash scripts/run_train_t1_ours_cw_eopd.sh

    echo "6/6: Running Permutation Control..."
    bash scripts/run_train_t1_permutation.sh
    echo "Table 1 experiments complete!"
}

cmd_table2() {
    echo ""
    echo ">>> [Step 5] Running Table 2 Ablation Experiments..."
    echo "1/3: Running Ablation T2(a) - {a_T} Only..."
    bash scripts/run_train_t2_ablation_at_only.sh

    echo "2/3: Running Ablation T2(b) - Length Normalized Allocation..."
    bash scripts/run_train_t2_ablation_length_norm.sh

    echo "3/3: Running Ablation T2(c) - KL Information Gain..."
    bash scripts/run_train_t2_ablation_kl.sh
    echo "Table 2 ablations complete!"
}

cmd_eval() {
    echo ""
    echo ">>> [Step 6] Running Benchmark Evaluations..."
    for exp_dir in checkpoints/*; do
        if [ -d "${exp_dir}" ]; then
            echo "Evaluating checkpoint in ${exp_dir}..."
            bash scripts/run_eval_all_benchmarks.sh "${exp_dir}" || echo "Warning: eval failed on ${exp_dir}"
        fi
    done
    echo "All evaluations complete!"
}

print_banner

STEP=${1:-"all"}

case "${STEP}" in
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
        cmd_env_check
        cmd_prepare_data
        cmd_cache_teacher
        cmd_phase0
        cmd_table1
        cmd_table2
        cmd_eval
        echo "\n======================================================================"
        echo "  ALL EXPERIMENTS FINISHED SUCCESSFULLY!"
        echo "======================================================================"
        ;;
    *)
        echo "Usage: bash scripts/run_all_experiments_master.sh [check|data|cache|phase0|table1|table2|eval|all]"
        exit 1
        ;;
esac
