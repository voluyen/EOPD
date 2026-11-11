#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (6): permutation control. g_k permuted among steps within strata of relative position and length.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
EXP_BASE="t1_permutation"
ensure_teacher_cache
run_training "$EXP_BASE" "CWOPDTrainer" "cw_opd" $(cw_overrides permutation)  "$@"
