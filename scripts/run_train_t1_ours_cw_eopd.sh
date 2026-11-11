#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (5): ours + EOPD. Weights also multiply EOPD's forward-KL term (tau=0.8, coef=1.0, top-16).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t1_ours_cw_eopd"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_eopd" $(cw_overrides none) "+actor_rollout_ref.ref.topk_logits=16" "$@"
