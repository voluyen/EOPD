#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (1): standard OPD (Lu & Thinking Machines Lab, 2025). Reverse-KL token advantage, no top-k soft KD.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_training "t1_opd_baseline" "OnPolicyDistillTrainer" "on_policy_distill"  "$@"
