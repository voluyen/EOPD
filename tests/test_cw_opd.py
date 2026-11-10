# Copyright 2026 The Consequence-Weighted OPD Authors.

"""Unit tests for Consequence-Weighted OPD components."""

import pytest
import torch
from types import SimpleNamespace

from verl.trainer.ppo.cw_opd.candidate_manager import (
    CandidateManager,
    canonicalize_answer,
    check_balanced_braces,
    extract_boxed_content,
    are_answers_equivalent,
)
from verl.trainer.ppo.cw_opd.probe_scorer import (
    StepBoundary,
    split_response_into_steps,
    compute_total_variation,
    compute_kl_divergence,
    permute_shifts_stratified,
    compute_consequence_weights,
    ProbeScorer,
)
from verl.trainer.ppo.cw_opd.policy_loss import (
    compute_policy_loss_cw_opd,
    compute_policy_loss_cw_eopd,
    compute_policy_loss_fuw,
)


def test_balanced_braces():
    assert check_balanced_braces("12")
    assert check_balanced_braces("\\frac{1}{2}")
    assert check_balanced_braces("{\\sqrt{5}}")
    assert not check_balanced_braces("12}")
    assert not check_balanced_braces("{\\frac{1}{2}")
    assert not check_balanced_braces("}")


def test_extract_boxed_content():
    text1 = "Therefore, the answer is \\boxed{42}."
    assert extract_boxed_content(text1) == "42"

    text2 = "First we find \\boxed{10}. Then final answer is \\boxed{\\frac{3}{4}}."
    assert extract_boxed_content(text2) == "\\frac{3}{4}"

    text_unbalanced = "Answer is \\boxed{42"
    assert extract_boxed_content(text_unbalanced) is None


def test_canonicalize_and_equivalence():
    assert canonicalize_answer(" $42$ ") == "42"
    assert canonicalize_answer(" \\text{5} ") == "5"
    assert are_answers_equivalent("5", "5.0")
    assert are_answers_equivalent("1/2", "0.5") or are_answers_equivalent("1/2", "1/2")
    assert not are_answers_equivalent("5", "7")


def test_candidate_manager():
    mgr = CandidateManager(max_candidates=3)
    p_id = "test_prompt_1"
    mgr.register_teacher_answer(p_id, "5")

    # Epoch 1: H_x is empty, C_x = {a_T}
    cands_ep1 = mgr.get_candidate_set(p_id)
    assert cands_ep1 == ["5"]

    # Student gives wrong answer in epoch 1
    mgr.update_history(p_id, "Step 1... \\boxed{7}")
    
    # Epoch 2: C_x = {5, 7}
    cands_ep2 = mgr.get_candidate_set(p_id)
    assert cands_ep2 == ["5", "7"]

    # Student gives another answer in epoch 2
    mgr.update_history(p_id, "Step 2... \\boxed{9}")

    # Epoch 3: C_x = {5, 7, 9}
    cands_ep3 = mgr.get_candidate_set(p_id)
    assert len(cands_ep3) == 3
    assert cands_ep3[0] == "5"

    # Ablation at_only returns only teacher answer
    cands_at_only = mgr.get_candidate_set(p_id, at_only=True)
    assert cands_at_only == ["5"]


def test_split_response_into_steps():
    resp = "Step 1: Calculate x.\nx = 2 + 3 = 5.\n\nStep 2: Calculate y.\ny = 5 * 2 = 10.\n\nStep 3: Conclude.\nThe answer is \\boxed{10}."
    steps = split_response_into_steps(resp)
    assert len(steps) >= 3
    assert steps[0].step_idx == 1
    assert steps[1].step_idx == 2
    assert steps[2].step_idx == 3


def test_total_variation_and_example():
    # Test worked example from Table 2 in proposal
    q0 = {"5": 0.60, "7": 0.10, "bot": 0.30}
    q1 = {"5": 0.62, "7": 0.09, "bot": 0.29}
    q2 = {"5": 0.15, "7": 0.55, "bot": 0.30}
    q3 = {"5": 0.14, "7": 0.57, "bot": 0.29}

    g1 = compute_total_variation(q1, q0)
    g2 = compute_total_variation(q2, q1)
    g3 = compute_total_variation(q3, q2)

    assert abs(g1 - 0.02) < 1e-4
    assert abs(g2 - 0.47) < 1e-4
    assert abs(g3 - 0.02) < 1e-4


def test_consequence_weights():
    # 2 rollouts in batch, length 6
    steps1 = [
        StepBoundary(1, 0, 2, 2, "s1"),
        StepBoundary(2, 2, 4, 2, "s2"),
        StepBoundary(3, 4, 6, 2, "s3"),
    ]
    shifts1 = [0.02, 0.47, 0.02]
    
    steps2 = [
        StepBoundary(1, 0, 3, 3, "s1"),
        StepBoundary(2, 3, 6, 3, "s2"),
    ]
    shifts2 = [0.10, 0.10]

    mask = torch.ones((2, 6), dtype=torch.float32)
    w_tilde, stats = compute_consequence_weights(
        batch_shifts=[shifts1, shifts2],
        batch_steps=[steps1, steps2],
        response_length=6,
        batch_size=2,
        response_mask=mask,
    )

    # Batch mean of w_tilde over valid tokens must be 1.0
    mean_w = w_tilde.mean().item()
    assert abs(mean_w - 1.0) < 1e-4
    # Step 2 in rollout 1 (index [0, 2:4]) must have significantly higher weight than Step 1
    assert w_tilde[0, 2].item() > w_tilde[0, 0].item()


def test_cw_opd_loss():
    bsz, resp_len = 2, 4
    old_log_prob = torch.zeros((bsz, resp_len), requires_grad=False)
    log_prob = torch.zeros((bsz, resp_len), requires_grad=True)
    ref_log_prob = torch.ones((bsz, resp_len), requires_grad=False) * 0.5
    advantages = torch.zeros((bsz, resp_len))
    response_mask = torch.ones((bsz, resp_len))
    cw_weights = torch.ones((bsz, resp_len)) * 1.5

    config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        global_batch_info={},
    )

    loss, metrics = compute_policy_loss_cw_opd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        ref_log_prob=ref_log_prob,
        cw_weights=cw_weights,
    )

    assert loss.requires_grad
    loss.backward()
    assert log_prob.grad is not None
    assert "actor/rev_kl" in metrics
    assert "actor/avg_weighted_advantage" in metrics


def test_fuw_loss():
    bsz, resp_len = 2, 4
    old_log_prob = torch.zeros((bsz, resp_len), requires_grad=False)
    log_prob = torch.zeros((bsz, resp_len), requires_grad=True)
    ref_log_prob = torch.ones((bsz, resp_len), requires_grad=False) * 0.5
    advantages = torch.zeros((bsz, resp_len))
    response_mask = torch.ones((bsz, resp_len))

    config = SimpleNamespace(
        clip_ratio=0.2,
        global_batch_info={},
    )

    loss, metrics = compute_policy_loss_fuw(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        ref_log_prob=ref_log_prob,
    )

    assert loss.requires_grad
    assert "actor/fuw_advantage" in metrics
