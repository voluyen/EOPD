#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Shared configuration + launcher for every training run (sourced by run_train_*.sh).
# All settings follow Section 6 of proposal v7 and can be overridden through env vars.
#
#   SEED=1 N_GPUS=8 bash scripts/run_train_t1_ours_cw_opd.sh        # env overrides
#   DRY_RUN=1 bash scripts/run_train_t1_eopd.sh                      # print the command only
#   bash scripts/run_train_t1_opd.sh trainer.total_epochs=1          # extra Hydra overrides pass through

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

# ---- models: prefer local copies made by scripts/download_models.sh -------------------
MODELS_DIR=${MODELS_DIR:-"${REPO_ROOT}/models"}
_resolve_model() { [ -d "${MODELS_DIR}/$(basename "$1")" ] && echo "${MODELS_DIR}/$(basename "$1")" || echo "$1"; }
STUDENT_MODEL=${STUDENT_MODEL:-$(_resolve_model "Qwen/Qwen3-1.7B-Base")}
TEACHER_MODEL=${TEACHER_MODEL:-$(_resolve_model "Qwen/Qwen3-8B")}

# ---- data / outputs --------------------------------------------------------------------
DATA_DIR=${DATA_DIR:-"data/math"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/train.parquet"}
VAL_FILE=${VAL_FILE:-"data/eval_benchmarks/math500.parquet"}
TEACHER_CACHE=${TEACHER_CACHE:-"${DATA_DIR}/teacher_answers_qwen3_8b.json"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"checkpoints"}
LOG_ROOT=${LOG_ROOT:-"logs"}
PROJECT_NAME=${PROJECT_NAME:-"CW_OPD_Experiments"}

# ---- hardware ---------------------------------------------------------------------------
N_GPUS=${N_GPUS:-4}
NNODES=${NNODES:-1}
ROLLOUT_GPU_UTIL=${ROLLOUT_GPU_UTIL:-0.5}   # student/teacher FSDP share the GPUs with vLLM
# Token budgets for dynamic batching (probe sequences are long, so this matters for CW-OPD)
ACTOR_MAX_TOKENS_PER_GPU=${ACTOR_MAX_TOKENS_PER_GPU:-24576}
LOGPROB_MAX_TOKENS_PER_GPU=${LOGPROB_MAX_TOKENS_PER_GPU:-49152}

# ---- training hyper-parameters (Section 6) ---------------------------------------------
LR=${LR:-3e-6}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-128}
TRAIN_PROMPT_MINI_BSZ=${TRAIN_PROMPT_MINI_BSZ:-32}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-1024}
MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
TEMPERATURE=${TEMPERATURE:-1.0}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-3}
SEED=${SEED:-0}

# ---- logging / checkpoints --------------------------------------------------------------
LOGGER=${LOGGER:-"[console,wandb,file]"}          # file => jsonl of every logged metric
SAVE_FREQ=${SAVE_FREQ:-50}                         # steps; save_freq=-1 would save NOTHING
TEST_FREQ=${TEST_FREQ:-25}                         # validation on MATH500 every N steps
KEEP_CKPTS=${KEEP_CKPTS:-3}
export VERL_FILE_LOGGER_ROOT=${VERL_FILE_LOGGER_ROOT:-"${REPO_ROOT}/${LOG_ROOT}/metrics"}
export WANDB_MODE=${WANDB_MODE:-online}            # WANDB_MODE=offline|disabled if no network / key
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1

# CW-OPD knobs
CW_MAX_STEPS=${CW_MAX_STEPS:-0}                    # cap on paragraphs per rollout (0 = none); lowers probe cost
CW_DUMP_EVERY=${CW_DUMP_EVERY:-10}                 # write per-rollout probe traces (jsonl) every N steps

require_file() {
  if [ ! -f "$1" ]; then echo "ERROR: $1 not found. $2" >&2; exit 1; fi
}

ensure_teacher_cache() {
  [ "${DRY_RUN:-0}" = "1" ] && return 0
  if [ ! -f "${TEACHER_CACHE}" ]; then
    echo "Teacher cache ${TEACHER_CACHE} missing -> generating (bash scripts/run_cache_teacher.sh)"
    DATA_PATH="${TRAIN_FILE}" OUTPUT_PATH="${TEACHER_CACHE}" TEACHER_MODEL="${TEACHER_MODEL}" \
      bash scripts/run_cache_teacher.sh
  fi
}

# run_training <exp_name> <trainer_class> <loss_mode> [extra hydra overrides...]
run_training() {
  local base_name=$1 trainer_class=$2 loss_mode=$3; shift 3
  local exp_name="${base_name}_seed${SEED}"
  local ckpt_dir="${OUTPUT_ROOT}/${exp_name}"
  local log_dir="${LOG_ROOT}/${exp_name}"
  local log_file="${log_dir}/train_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${log_dir}" "${VERL_FILE_LOGGER_ROOT}"

  local cmd=(python3 -m verl.trainer.main_ppo
    "+trainer.trainer_class=${trainer_class}"
    "algorithm.adv_estimator=on_policy"
    "actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}"
    "+actor_rollout_ref.teacher_model.path=${TEACHER_MODEL}"
    "actor_rollout_ref.model.path=${STUDENT_MODEL}"
    "data.train_files=${TRAIN_FILE}" "data.val_files=${VAL_FILE}"
    "data.train_batch_size=${TRAIN_PROMPT_BSZ}"
    "data.max_prompt_length=${MAX_PROMPT_LEN}" "data.max_response_length=${MAX_RESP_LEN}"
    "data.seed=${SEED}"
    "actor_rollout_ref.rollout.name=vllm" "actor_rollout_ref.rollout.n=1"
    "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_UTIL}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
    "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKENS_PER_GPU}"
    "actor_rollout_ref.actor.optim.lr=${LR}"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_PROMPT_MINI_BSZ}"
    "actor_rollout_ref.actor.ppo_epochs=1"
    "actor_rollout_ref.actor.use_dynamic_bsz=True"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKENS_PER_GPU}"
    "actor_rollout_ref.actor.use_kl_loss=False" "algorithm.use_kl_in_reward=False"
    "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${LOGPROB_MAX_TOKENS_PER_GPU}"
    "actor_rollout_ref.ref.fsdp_config.param_offload=True"
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,hf_model,extra]"
    "trainer.use_legacy_worker_impl=enable"
    "trainer.project_name=${PROJECT_NAME}" "trainer.experiment_name=${exp_name}"
    "trainer.logger=${LOGGER}"
    "trainer.n_gpus_per_node=${N_GPUS}" "trainer.nnodes=${NNODES}"
    "trainer.total_epochs=${TOTAL_EPOCHS}"
    "trainer.val_before_train=True" "trainer.test_freq=${TEST_FREQ}"
    "trainer.save_freq=${SAVE_FREQ}" "trainer.max_actor_ckpt_to_keep=${KEEP_CKPTS}"
    "trainer.log_val_generations=8"
    "trainer.default_local_dir=${ckpt_dir}"
    "$@")

  echo "=========================================================="
  echo "Experiment : ${exp_name}   (trainer=${trainer_class}, loss=${loss_mode})"
  echo "Student    : ${STUDENT_MODEL}"
  echo "Teacher    : ${TEACHER_MODEL}"
  echo "GPUs       : ${N_GPUS} x ${NNODES}   seed=${SEED}"
  echo "Checkpoints: ${ckpt_dir}"
  echo "Log file   : ${log_file}"
  echo "Metrics    : ${VERL_FILE_LOGGER_ROOT}/${PROJECT_NAME}/${exp_name}.jsonl  (+ wandb: ${LOGGER})"
  echo "=========================================================="
  if [ "${DRY_RUN:-0}" = "1" ]; then printf '%s\n' "${cmd[@]}"; return 0; fi

  # git revision + resolved command go next to the log for reproducibility
  { echo "# git: $(git rev-parse HEAD 2>/dev/null || echo unknown) dirty=$(git status --porcelain 2>/dev/null | wc -l)"
    printf '%q ' "${cmd[@]}"; echo; } > "${log_dir}/command.txt"
  "${cmd[@]}" 2>&1 | tee "${log_file}"
}

# CW-OPD overrides (cw_opd.* is not declared in the yaml, hence the '+')
cw_overrides() {   # <ablation_mode>
  echo "+cw_opd.teacher_cache_path=${TEACHER_CACHE}" "+cw_opd.ablation_mode=$1" "+cw_opd.seed=${SEED}" \
       "+cw_opd.max_steps=${CW_MAX_STEPS}" "+cw_opd.dump_every=${CW_DUMP_EVERY}" \
       "+cw_opd.log_dir=${LOG_ROOT}/${EXP_BASE:-cw}_seed${SEED}/cw_opd"
}
