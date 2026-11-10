#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 - Experiment 3: FUW Baseline (Token utility via local trust-region, 2026)
# Setup: Qwen3-8B teacher, Qwen3-1.7B-Base student, MATH split, loss_mode=fuw

set -euo pipefail

PROJECT_NAME=${PROJECT_NAME:-"CW_OPD_Experiments"}
EXP_NAME="t1_fuw_baseline"

# Model paths
STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3-1.7B-Base"}
TEACHER_MODEL=${TEACHER_MODEL:-"Qwen/Qwen3-8B"}

# Data & Checkpoint paths
TRAIN_FILE=${TRAIN_FILE:-"data/math/train.parquet"}
TEST_FILE=${TEST_FILE:-"data/eval_benchmarks/math500.parquet"}
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
echo "Starting Experiment: Table 1 (3) - FUW Baseline"
echo "Student:    ${STUDENT_MODEL}"
echo "Teacher:    ${TEACHER_MODEL}"
echo "Loss Mode:  fuw (trust-region token utility)"
echo "GPUs:       ${N_GPUS} x ${NNODES} node(s)"
echo "Output:     ${CKPTS_DIR}"
echo "=========================================================="

python3 -m verl.trainer.main_ppo \
    trainer.trainer_class="OnPolicyDistillTrainer" \
    algorithm.adv_estimator="on_policy" \
    actor_rollout_ref.actor.policy_loss.loss_mode="fuw" \
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
