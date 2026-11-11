#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 2 (a): C_x={a_T} only, g_k=|q_k(a_T)-q_{k-1}(a_T)| (contribution of the answer history).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t2_at_only"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_opd" $(cw_overrides at_only)  "$@"
