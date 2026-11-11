#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Evaluate one trained run (or any HF model dir) on MATH500, AIME24, AIME25, AMC23, Minerva, OlympiadBench.
# Protocol: temp 1.0, top-p 0.8, 8192 tokens, 8 samples -> Avg@8 / Pass@8 (vLLM).
#
#   bash scripts/run_eval_all_benchmarks.sh checkpoints/t1_ours_cw_opd_seed0     # a training run dir
#   bash scripts/run_eval_all_benchmarks.sh models/Qwen3-1.7B-Base               # the untrained student
#
# A run dir is resolved to its latest global_step_N/actor/huggingface; if only sharded FSDP weights were
# saved they are merged first. Results: <run dir>/eval_results.json (+ .per_problem.json).

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TARGET=${1:-${MODEL_PATH:-}}
[ -n "${TARGET}" ] || { echo "usage: $0 <run_dir | hf_model_dir>"; exit 1; }
BENCHMARKS_DIR=${BENCHMARKS_DIR:-"data/eval_benchmarks"}
N_SAMPLES=${N_SAMPLES:-8}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.8}
MAX_TOKENS=${MAX_TOKENS:-8192}
EVAL_SEED=${EVAL_SEED:-0}
TP=${TP:-1}

if [ -f "${TARGET}/config.json" ]; then            # already an HF directory
  HF_DIR="${TARGET}"
  OUTPUT_PATH=${OUTPUT_PATH:-"${TARGET}/eval_results.json"}
else                                                # training run directory
  [ -f "${TARGET}/latest_checkpointed_iteration.txt" ] || { echo "ERROR: no checkpoint in ${TARGET}"; exit 1; }
  STEP=$(cat "${TARGET}/latest_checkpointed_iteration.txt")
  ACTOR_DIR="${TARGET}/global_step_${STEP}/actor"
  HF_DIR="${ACTOR_DIR}/huggingface"
  if [ ! -f "${HF_DIR}/model.safetensors" ] && ! ls "${HF_DIR}"/*.safetensors >/dev/null 2>&1; then
    HF_DIR="${TARGET}/global_step_${STEP}/merged_hf"
    if [ ! -f "${HF_DIR}/config.json" ]; then
      echo "Merging FSDP shards of step ${STEP} -> ${HF_DIR}"
      python3 -m verl.model_merger merge --backend fsdp --local_dir "${ACTOR_DIR}" --target_dir "${HF_DIR}"
    fi
  fi
  OUTPUT_PATH=${OUTPUT_PATH:-"${TARGET}/eval_results.json"}
  echo "Run ${TARGET}: step ${STEP}"
fi

if [ -z "$(ls "${BENCHMARKS_DIR}"/*.parquet 2>/dev/null)" ]; then
  echo "Benchmarks missing in ${BENCHMARKS_DIR} -> preparing"
  python3 scripts/prepare_eval_benchmarks.py --output_dir "${BENCHMARKS_DIR}"
fi

echo "Evaluating ${HF_DIR} -> ${OUTPUT_PATH}"
python3 eval/evaluate_benchmarks.py \
  --model_path "${HF_DIR}" --benchmarks_dir "${BENCHMARKS_DIR}" --output_path "${OUTPUT_PATH}" \
  --n_samples "${N_SAMPLES}" --temperature "${TEMPERATURE}" --top_p "${TOP_P}" \
  --max_tokens "${MAX_TOKENS}" --seed "${EVAL_SEED}" --tp "${TP}" 2>&1 | tee "${OUTPUT_PATH%.json}.log"
