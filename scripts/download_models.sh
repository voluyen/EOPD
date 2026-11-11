#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Download teacher + student weights into ./models (scripts/common.sh picks them up automatically).
#   HF_TOKEN=... bash scripts/download_models.sh
#   MODELS="Qwen/Qwen3-8B Qwen/Qwen3-1.7B-Base" MODELS_DIR=/data/models bash scripts/download_models.sh
# Optional students for the conditional extensions: MODELS="... Qwen/Qwen3-0.6B-Base Qwen/Qwen3-4B-Base"

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

MODELS_DIR=${MODELS_DIR:-models}
MODELS=${MODELS:-"Qwen/Qwen3-8B Qwen/Qwen3-1.7B-Base"}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
mkdir -p "${MODELS_DIR}"

for repo in ${MODELS}; do
  target="${MODELS_DIR}/$(basename "${repo}")"
  if [ -f "${target}/config.json" ] && [ -f "${target}/.download_complete" ]; then
    echo "[skip] ${repo} already in ${target}"; continue
  fi
  echo "[download] ${repo} -> ${target}"
  python3 - "${repo}" "${target}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, target = sys.argv[1:3]
snapshot_download(repo_id=repo, local_dir=target, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
open(f"{target}/.download_complete", "w").write("ok")
PY
done

echo "Verifying tokenizers / configs..."
python3 - "${MODELS_DIR}" ${MODELS} <<'PY'
import os, sys
from transformers import AutoConfig, AutoTokenizer
root, repos = sys.argv[1], sys.argv[2:]
toks = {}
for r in repos:
    p = os.path.join(root, os.path.basename(r))
    cfg = AutoConfig.from_pretrained(p); tok = AutoTokenizer.from_pretrained(p)
    toks[r] = tok
    print(f"{r}: {cfg.model_type}, vocab={len(tok)}, eos={tok.eos_token!r}, chat_template={'yes' if tok.chat_template else 'NO'}")
ref = next(iter(toks.values()))
for r, t in toks.items():
    probe = "Therefore, the final answer is \\boxed{12}"
    assert t.encode(probe) == ref.encode(probe), f"tokenizer of {r} differs: probes would not be comparable"
print("tokenizers agree on the probe string")
PY
