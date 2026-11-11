#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 2 (c): information-gain variant KL(q_k||q_{k-1}) instead of TV.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t2_kl"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_opd" $(cw_overrides kl_variant)  "$@"
