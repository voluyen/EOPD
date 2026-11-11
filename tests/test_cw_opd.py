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


# ----------------------------------------------------------------------------------------
# Token-exact probing (teacher_probe.py) with a fake char-level tokenizer and fake teacher
# ----------------------------------------------------------------------------------------
from verl.trainer.ppo.cw_opd.candidate_manager import (  # noqa: E402
    extract_question_from_chat_prompt,
    get_prompt_hash,
    question_text_from_messages,
)
from verl.trainer.ppo.cw_opd.probe_scorer import shifts_from_beliefs  # noqa: E402
from verl.trainer.ppo.cw_opd.teacher_probe import (  # noqa: E402
    BOT_KEY,
    probe_batch,
    segment_token_ids,
    tokenize_probe_and_candidate,
)


class CharTokenizer:
    """One token per character (so '\\n\\n' is two tokens: exercises cross-token separator detection)."""

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids if i != 0)


def fake_teacher(tok):
    """Believes 5 unless the context contains 'BAD', then believes 7. Mass 0.8/0.05 on the listed candidates."""

    def score_fn(input_ids, attention_mask, position_ids, responses):
        out = torch.full(responses.shape, 0.0)
        for b in range(input_ids.shape[0]):
            row = input_ids[b][attention_mask[b].bool()].tolist()
            n_resp = int((responses[b] != 0).sum().item())
            ctx, cand = tok.decode(row[: len(row) - n_resp]), tok.decode(row[len(row) - n_resp:])
            fav = "7}" if "BAD" in ctx else "5}"
            p = 0.8 if cand == fav else 0.05
            out[b, 0] = float(torch.log(torch.tensor(p)))
        return out

    return score_fn


def test_segment_token_ids_exact_alignment():
    tok = CharTokenizer()
    text = "Step one.\n\nStep two BAD.\n\n\nStep three \\boxed{7}"
    ids = tok.encode(text)
    steps = segment_token_ids(ids, tok)
    assert len(steps) == 3
    # the steps tile the response exactly, with no gaps and no overlap
    assert steps[0].token_start == 0 and steps[-1].token_end == len(ids)
    for a, b in zip(steps, steps[1:]):
        assert a.token_end == b.token_start
    assert "".join(s.text for s in steps) == text
    assert steps[0].text == "Step one.\n\n"


def test_segment_falls_back_to_single_newline_and_caps_steps():
    tok = CharTokenizer()
    ids = tok.encode("a\nb\nc\nd\ne")
    assert len(segment_token_ids(ids, tok)) == 5
    assert len(segment_token_ids(ids, tok, max_steps=2)) == 2
    assert segment_token_ids(ids, tok, max_steps=2)[-1].token_end == len(ids)


def test_probe_batch_detects_belief_shift_at_the_right_step():
    tok = CharTokenizer()
    resp = "Step one.\n\nStep two BAD.\n\nStep three."
    prompt = "Q?"
    probes, stats = probe_batch(
        prompt_ids_list=[tok.encode(prompt)],
        resp_ids_list=[tok.encode(resp) + [0]],  # trailing EOS/pad must not enter the context
        candidates_list=[["5", "7"]],
        tokenizer=tok,
        score_fn=fake_teacher(tok),
        pad_token_id=0,
        strip_trailing_ids=[0],
    )
    pr = probes[0]
    assert len(pr.beliefs) == len(pr.steps) + 1 == 4
    # q_0, q_1: teacher says 5 (0.8), 7 (0.05), residual 0.15
    assert abs(pr.beliefs[0]["5"] - 0.8) < 1e-5 and abs(pr.beliefs[0][BOT_KEY] - 0.15) < 1e-5
    # after step 2 the context contains BAD: the belief flips
    assert abs(pr.beliefs[2]["7"] - 0.8) < 1e-5
    g = shifts_from_beliefs(pr.beliefs, pr.steps)
    assert g[0] < 1e-6 and g[2] < 1e-6
    assert abs(g[1] - 0.75) < 1e-5  # 0.5 * (|0.05-0.8| + |0.8-0.05| + 0)
    assert stats["probe_sequences"] == 4 * 2


def test_probe_with_no_candidates_gives_zero_shift_not_a_dummy():
    tok = CharTokenizer()
    probes, _ = probe_batch(
        [tok.encode("Q")], [tok.encode("a\n\nb")], [[]], tok, fake_teacher(tok), pad_token_id=0
    )
    assert shifts_from_beliefs(probes[0].beliefs, probes[0].steps) == [0.0, 0.0]


def test_probe_scorer_refuses_silent_mock():
    scorer = ProbeScorer()
    with pytest.raises(ValueError):
        scorer.compute_rollout_shifts("Q", "a\n\nb", ["5"], teacher_model=None, tokenizer=None)


def test_tokenize_probe_and_candidate_splits_joint_tokenization():
    tok = CharTokenizer()
    probe_ids, cand_ids, retok = tokenize_probe_and_candidate(tok, "12")
    assert tok.decode(cand_ids) == "12}" and not retok
    assert tok.decode(probe_ids).endswith("\\boxed{")


def test_prompt_key_matches_between_cache_and_trainer():
    raw = "What is $2+3$?  Let's think step by step and output the final answer within \\boxed{}."
    dataset_key = get_prompt_hash(question_text_from_messages([{"role": "user", "content": raw}]))
    templated = f"<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n{raw}<|im_end|>\n<|im_start|>assistant\n"
    assert get_prompt_hash(extract_question_from_chat_prompt(templated)) == dataset_key


def test_shift_variants_from_beliefs():
    steps = [StepBoundary(1, 0, 2, 2, "a"), StepBoundary(2, 2, 6, 4, "b")]
    beliefs = [{"5": 0.6, BOT_KEY: 0.4}, {"5": 0.6, BOT_KEY: 0.4}, {"5": 0.1, BOT_KEY: 0.9}]
    assert shifts_from_beliefs(beliefs, steps) == [0.0, pytest.approx(0.5)]
    assert shifts_from_beliefs(beliefs, steps, "length_norm")[1] == pytest.approx(0.5 / 4)
    assert shifts_from_beliefs(beliefs, steps, "kl_variant")[1] > 0


def test_trainer_end_to_end_cpu(tmp_path):
    """Drive CWOPDTrainer._compute_ref_log_prob's CW part with a fake teacher and a hand-built batch."""
    pytest.importorskip("torchdata")
    from verl import DataProto
    from verl.trainer.ppo.cw_opd.cw_opd_trainer import CWOPDTrainer

    tok = CharTokenizer()
    question = "What is 2+3?"
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    resp = "Step one.\n\nStep two BAD.\n\nStep three."

    t = object.__new__(CWOPDTrainer)
    t.tokenizer = tok
    t.candidate_manager = CandidateManager()
    t.candidate_manager.register_teacher_answer(get_prompt_hash(question), "5")
    t.candidate_manager.update_history_answer(get_prompt_hash(question), "7")  # earlier-epoch answer
    t.ablation_mode, t.max_steps, t.probe_chunk_size, t.dump_every, t.seed = "none", 0, 8, 1, 0
    t.log_dir, t.history_path = str(tmp_path), str(tmp_path / "h.json")
    t._extra_metrics, t._cum_probe_tokens, t._cum_probe_time, t.global_steps = {}, 0.0, 0.0, 1
    t.ref_policy_wg = SimpleNamespace(world_size=1)
    fake = fake_teacher(tok)
    t._teacher_score_fn = lambda *a: fake(*a)

    p_ids, r_ids = tok.encode(prompt), tok.encode(resp)
    P, R = len(p_ids) + 3, len(r_ids) + 5  # left-pad prompt, right-pad response
    prompts = torch.tensor([[0] * 3 + p_ids])
    responses = torch.tensor([r_ids + [0] * 5])
    mask = torch.tensor([[0] * 3 + [1] * len(p_ids) + [1] * len(r_ids) + [0] * 5])
    batch = DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "attention_mask": mask,
            "response_mask": mask[:, P:],
            "input_ids": torch.cat([prompts, responses], dim=1),
        }
    )
    w, metrics = t._compute_consequence_weights(batch)

    assert w.shape == responses.shape
    assert abs(w[mask[:, P:].bool()].mean().item() - 1.0) < 1e-4
    # the step that introduces BAD (step 2) must carry the largest weight
    steps_end = [len("Step one.\n\n"), len("Step one.\n\nStep two BAD.\n\n"), len(resp)]
    assert w[0, steps_end[0] + 1].item() > w[0, 1].item()
    assert w[0, steps_end[0] + 1].item() > w[0, steps_end[1] + 1].item()
    assert metrics["cw/aT_cache_miss_frac"] == 0.0
    assert metrics["cw/n_candidates_mean"] == 2.0
    assert metrics["cw/g_tv_max"] == pytest.approx(0.75, abs=1e-4)
    assert metrics["cw/probe_sequences"] == 4 * 2
    # trace file written and the rollout's own answer was added to H_x only after weighting
    assert (tmp_path / "probe_step000001.jsonl").exists()
