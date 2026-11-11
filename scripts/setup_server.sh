#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# One-time server setup: python env, CUDA stack, verl (this repo), output dirs, W&B.
# Pinned to the versions of docker/verl0.6-cu128-torch2.8.0-fa2.7.4 (override via env vars).
# Alternative: use the Docker images in docker/ and only run `pip install -e .` inside them.
#
#   bash scripts/setup_server.sh              # create env "cwopd" + install everything
#   ENV_NAME=myenv PYTHON_VERSION=3.11 bash scripts/setup_server.sh
#   SKIP_ENV=1 bash scripts/setup_server.sh   # already inside the right env: only install packages

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ENV_NAME=${ENV_NAME:-cwopd}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}
TORCH_VERSION=${TORCH_VERSION:-2.8.0}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}
VLLM_VERSION=${VLLM_VERSION:-0.11.0}
FLASH_ATTN_VERSION=${FLASH_ATTN_VERSION:-2.7.4.post1}
N_GPUS=${N_GPUS:-4}

echo ">>> [1/6] Hardware check"
if ! command -v nvidia-smi >/dev/null 2>&1; then echo "ERROR: nvidia-smi not found (no NVIDIA driver)"; exit 1; fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_COUNT=$(nvidia-smi -L | grep -c '^GPU')
[ "${GPU_COUNT}" -ge "${N_GPUS}" ] || echo "WARNING: ${GPU_COUNT} GPUs visible but N_GPUS=${N_GPUS}"
echo "Free disk (need ~100GB for models + checkpoints):"; df -h . | tail -1

echo ">>> [2/6] Python environment"
if [ "${SKIP_ENV:-0}" != "1" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda env list | grep -q "^${ENV_NAME} " || conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
    conda activate "${ENV_NAME}"
  else
    python3 -m venv ".venv-${ENV_NAME}"
    # shellcheck disable=SC1091
    source ".venv-${ENV_NAME}/bin/activate"
  fi
fi
python --version
pip install -U pip wheel setuptools ninja packaging

echo ">>> [3/6] PyTorch ${TORCH_VERSION}, vLLM ${VLLM_VERSION}, flash-attn ${FLASH_ATTN_VERSION}"
pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
pip install "vllm==${VLLM_VERSION}"      # vLLM may pin its own torch; re-check below
pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

echo ">>> [4/6] verl (this repo) + CW-OPD extras"
pip install -r requirements.txt
pip install -e . --no-deps
pip install scipy math-verify huggingface_hub hf_transfer wandb pytest

echo ">>> [5/6] Directories"
mkdir -p models data/math data/eval_benchmarks checkpoints logs experiments

echo ">>> [6/6] Verification"
python - <<'PY'
import torch, importlib
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for m in ("vllm", "flash_attn", "verl", "ray", "math_verify", "wandb"):
    try:
        mod = importlib.import_module(m); print(f"{m:12s} OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(f"{m:12s} FAILED: {e}")
PY
pytest tests/test_cw_opd.py -q

cat <<MSG

Setup done. Next:
  export HF_TOKEN=...            # if the models need authentication
  wandb login                    # or: export WANDB_MODE=offline  /  WANDB_MODE=disabled
  bash scripts/download_models.sh
  bash scripts/run_all_experiments_master.sh data
MSG
