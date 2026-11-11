#!/usr/bin/env bash
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Table 1 (3): FUW token-utility reweighting baseline.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_training "t1_fuw" "OnPolicyDistillTrainer" "fuw"  "$@"
