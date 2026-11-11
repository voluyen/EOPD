#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 2 (b): length-normalised allocation g_k/L_k (separates consequence from paragraph length).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t2_length_norm"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_opd" $(cw_overrides length_norm)  "$@"
