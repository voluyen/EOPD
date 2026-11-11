#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (2): EOPD (Jin et al., ICML 2026). Entropy-gated forward KL: tau=0.8, coef=1.0 (loss defaults), top-16 teacher logits.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_training "t1_eopd" "OnPolicyDistillTrainer" "on_policy_distill" "+actor_rollout_ref.ref.topk_logits=16" "$@"
