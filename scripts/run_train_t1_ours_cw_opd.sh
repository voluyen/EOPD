#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (4): ours. Answer-shift-weighted OPD (Algorithm 1): C_x={a_T} U H_x, g_k=TV(q_k,q_{k-1}), w_k=1+g_k/g_bar.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t1_ours_cw_opd"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_opd" $(cw_overrides none)  "$@"
