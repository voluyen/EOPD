#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 2 - Ablation (b): Length-Normalized Allocation g_k / L_k
# Section 6: "length-normalized allocation g_k / L_k, with the same floor
# and batch normalization, to separate step consequence from paragraph length"

set -euo pipefail

PROJECT_NAME=${PROJECT_NAME:-"CW_OPD_Experiments"}
EXP_NAME="t2_ablation_length_norm"

# Model paths
STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3-1.7B-Base"}
TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen3-8B"}

# Data & Checkpoint paths
TRAIN_FILE=${TRAIN_FILE:-"data/math/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/eval_benchmarks/math500.parquet"}
TEACHER_CACHE=${TEACHER_CACHE:-"data/math/teacher_answers_qwen3_8b.json"}
CKPTS_DIR=${CKPTS_DIR:-"checkpoints/${EXP_NAME}"}

# Hardware settings
N_GPUS=${N_GPUS:-4}
NNODES=${NNODES:-1}

# Training Hyperparameters
LR=3e-6
TRAIN_PROMPT_BSZ=128
TRAIN_PROMPT_MINI_BSZ=32
MAX_PROMPT_LEN=1024
MAX_RESP_LEN=4096
TEMPERATURE=1.0
TOTAL_EPOCHS=3

echo "=========================================================="
echo "Starting Experiment: Table 2 (b) - Length-Normalized Shift"
echo "Student:       ${STUDENT_MODEL}"
echo "Teacher:       ${TEACHER_MODEL}"
echo "Ablation Mode: length_norm (g_k / L_k token allocation)"
echo "GPUs:          ${N_GPUS} x ${NNODES} node(s)"
echo "Output:        ${CKPTS_DIR}"
echo "=========================================================="

python3 -m verl.trainer.main_ppo \
    trainer.trainer_class="CWOPDTrainer" \
    algorithm.adv_estimator="on_policy" \
    actor_rollout_ref.actor.policy_loss.loss_mode="cw_opd" \
    cw_opd.teacher_cache_path="${TEACHER_CACHE}" \
    cw_opd.ablation_mode="length_norm" \
    actor_rollout_ref.teacher_model.path="${TEACHER_MODEL}" \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${TRAIN_PROMPT_BSZ} \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESP_LEN} \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ} \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir="${CKPTS_DIR}"
