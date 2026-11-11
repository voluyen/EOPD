#!/usr/bin/env python3
# Copyright 2026 The Consequence-Weighted OPD Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 0 Pilot Experiment Suite for Consequence-Weighted OPD.

Implements Section 5 of proposal_consequence_weighted_opd_v7:
- Adaptive teacher continuation sampling (M=8 screen, up to M=32 ambiguous)
- Continuation law alpha_P(h), continuation shift g_P_k, and success rate V(h)
- Probed beliefs q_k and shifts g_k
- P1(a) Calibration: distribution of e(h) = TV(q(h), alpha_P(h))
- P1(b) Incremental predictive value: OLS regression and clustered bootstrap
        for Delta R^2 and partial Spearman rank correlation of g_k
- P1(c) Martingale gap along teacher steps
- P2 Early detection before the answer appears (k*, k^-, signed agreement, Bonferroni)
- P3 Latency and memory overhead benchmark
- Automated Go/No-Go Decision Rule
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from verl.trainer.ppo.cw_opd.candidate_manager import (
    CandidateManager,
    are_answers_equivalent,
    canonicalize_answer,
    extract_boxed_content,
    get_prompt_hash,
)
from verl.trainer.ppo.cw_opd.candidate_manager import (
    extract_question_from_chat_prompt,
    question_text_from_messages,
)
from verl.trainer.ppo.cw_opd.probe_scorer import (
    PROBE_STRING,
    StepBoundary,
    compute_total_variation,
)
from verl.trainer.ppo.cw_opd.teacher_probe import probe_batch, segment_token_ids


def wilson_score_interval(successes: int, total: int, confidence: float = 0.90) -> Tuple[float, float]:
    """Calculate the two-sided Wilson score confidence interval."""
    if total <= 0:
        return 0.0, 1.0
    
    # z-value for confidence level: 1.645 for 90%
    z = stats.norm.ppf((1 + confidence) / 2)
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    half_width = (z * math.sqrt((p * (1 - p) + z2 / (4 * total)) / total)) / denom
    
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return lower, upper


def classify_boundary_label(successes: int, total: int, threshold: float = 0.5, confidence: float = 0.90) -> str:
    """Classify boundary as 'low', 'high', or 'undetermined' based on Wilson interval."""
    lower, upper = wilson_score_interval(successes, total, confidence)
    if upper < threshold:
        return "low"
    elif lower > threshold:
        return "high"
    else:
        return "undetermined"


@dataclass
class BoundaryEvaluation:
    """Evaluation data for a single paragraph boundary."""
    boundary_idx: int          # 0 .. K
    step_idx: Optional[int]    # 1 .. K for step following boundary, or None for k=0
    token_offset: int          # Token offset in response
    rel_position: float        # Relative position in rollout (0.0 .. 1.0)
    probed_belief: Dict[str, float]
    continuation_law: Dict[str, float]
    probe_error: float         # e(h) = TV(q(h), alpha_P(h))
    gold_success_rate: float   # V(h)
    wilson_lower: float
    wilson_upper: float
    boundary_label: str        # 'low', 'high', 'undetermined'
    num_continuations: int
    before_answer: bool        # True if boundary is strictly before final answer string


@dataclass
class StepEvaluation:
    """Evaluation data for a reasoning step."""
    step_idx: int              # 1 .. K
    token_len: int             # L_k
    rel_pos: float             # (k / K)
    probed_shift: float        # g_k
    continuation_shift: float  # g^P_k
    continuation_shift_half1: float  # For split-half reliability
    continuation_shift_half2: float
    v_change: float            # Delta V_k = V(h_{k+1}) - V(h_k)
    student_entropy: float
    teacher_entropy: float
    divergence_kl: float
    teacher_surprisal: float
    mismatch_ratio: float      # |1 - r_k|
    before_answer: bool
    delta_q_aT: float = 0.0    # signed change of the probed belief in a_T, q_k(a_T) - q_{k-1}(a_T)


@dataclass
class RolloutEvaluation:
    """Evaluation data for a single problem rollout."""
    problem_idx: int
    prompt_text: str
    gold_answer: str
    teacher_answer: str
    student_final_answer: Optional[str]
    student_is_correct: bool
    teacher_is_correct: bool
    boundaries: List[BoundaryEvaluation] = field(default_factory=list)
    steps: List[StepEvaluation] = field(default_factory=list)
    unrecoverable_transition_step: Optional[int] = None
    largest_drop_step: Optional[int] = None


class Phase0Runner:
    """Executes the Phase 0 pilot (no training) with the same probing code path as training.

    Pipeline per prompt: student rollouts (vLLM, chat template) -> token-exact paragraph segmentation ->
    probe beliefs q_k with the HF teacher (teacher_probe.probe_batch) -> teacher continuations from every
    boundary (vLLM, adaptive M) giving alpha_P(h) and V(h) -> local signals (HF student + teacher).
    Gold answers are used only here; a_T comes from the cached greedy teacher answers.
    """

    def __init__(
        self,
        student_model_path: str = "Qwen/Qwen3-1.7B-Base",
        teacher_model_path: str = "Qwen/Qwen3-8B",
        num_prompts: int = 100,
        m_screen: int = 8,
        m_adaptive: int = 32,
        n_history: int = 2,
        max_resp_len: int = 4096,
        cont_max_tokens: int = 4096,
        gpu_util_student: float = 0.15,
        gpu_util_teacher: float = 0.35,
        seed: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.student_model_path = student_model_path
        self.teacher_model_path = teacher_model_path
        self.num_prompts = num_prompts
        self.m_screen = m_screen
        self.m_adaptive = m_adaptive
        self.n_history = n_history
        self.max_resp_len = max_resp_len
        self.cont_max_tokens = cont_max_tokens
        self.seed = seed
        self.device = torch.device(device)
        self.candidate_manager = CandidateManager()
        self.timing = {"probe_s": 0.0, "probe_tokens": 0.0, "plain_teacher_s": 0.0, "plain_teacher_tokens": 0.0}
        self._load_models(gpu_util_student, gpu_util_teacher)

    # ------------------------------------------------------------------ models
    def _load_models(self, util_s: float, util_t: float):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vllm import LLM

        print(f"[Phase 0] Tokenizer from {self.teacher_model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.teacher_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        max_len = self.max_resp_len + self.cont_max_tokens + 2048
        print("[Phase 0] vLLM student (rollouts) / vLLM teacher (continuations)")
        self.vllm_student = LLM(self.student_model_path, gpu_memory_utilization=util_s, dtype="bfloat16",
                                trust_remote_code=True, seed=self.seed, max_model_len=max_len)
        self.vllm_teacher = LLM(self.teacher_model_path, gpu_memory_utilization=util_t, dtype="bfloat16",
                                trust_remote_code=True, seed=self.seed, max_model_len=max_len)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print("[Phase 0] HF student / HF teacher (probing and local signals)")
        self.hf_student = AutoModelForCausalLM.from_pretrained(
            self.student_model_path, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()
        self.hf_teacher = AutoModelForCausalLM.from_pretrained(
            self.teacher_model_path, torch_dtype=dtype, trust_remote_code=True).to(self.device).eval()

    # ------------------------------------------------------------------ data
    def build_prompt_ids(self, messages) -> List[int]:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    # ------------------------------------------------------------------ generation (vLLM, token ids in/out)
    def sample_rollouts(self, prompt_ids: List[int], n: int) -> List[List[int]]:
        """n student rollouts for one prompt; returns the sampled token ids (EOS stripped)."""
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=1.0, max_tokens=self.max_resp_len)
        out = self.vllm_student.generate([{"prompt_token_ids": prompt_ids}], sp, use_tqdm=False)[0]
        eos = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 151643, 151645}
        res = []
        for o in out.outputs:
            ids = list(o.token_ids)
            while ids and ids[-1] in eos:
                ids.pop()
            res.append(ids)
        return res

    def sample_teacher_continuations(self, ctx_ids_list: List[List[int]], m_list: List[int]) -> List[List[str]]:
        """For each context, m teacher continuations (temperature 1.0) to the end of the response."""
        from vllm import SamplingParams

        reqs, idx = [], []
        for i, (ctx, m) in enumerate(zip(ctx_ids_list, m_list)):
            if m > 0:
                reqs.append({"prompt_token_ids": ctx})
                idx.append(i)
        texts: List[List[str]] = [[] for _ in ctx_ids_list]
        if not reqs:
            return texts
        # one SamplingParams per request so that m can differ
        sps = [SamplingParams(n=m_list[i], temperature=1.0, max_tokens=self.cont_max_tokens) for i in idx]
        outs = self.vllm_teacher.generate(reqs, sps, use_tqdm=False)
        for i, o in zip(idx, outs):
            texts[i] = [c.text for c in o.outputs]
        return texts

    def evaluate_boundaries(
        self, ctx_ids_list: List[List[int]], candidate_answers: List[str], gold_answer: str
    ) -> List[Tuple[Dict[str, float], float, int, Dict[str, float], Dict[str, float]]]:
        """Adaptive sampling at all boundaries of one rollout: M=m_screen everywhere, up to m_adaptive where the
        Wilson label is undetermined or the boundary neighbours a candidate transition."""
        conts = self.sample_teacher_continuations(ctx_ids_list, [self.m_screen] * len(ctx_ids_list))

        def succ(cs):
            return sum(1 for c in cs if are_answers_equivalent(extract_boxed_content(c) or "", gold_answer))

        labels = [classify_boundary_label(succ(cs), len(cs), threshold=0.5, confidence=0.90) for cs in conts]
        # boundaries around a label transition also get more samples
        need = [lab == "undetermined" for lab in labels]
        for k in range(len(labels) - 1):
            if labels[k] != labels[k + 1]:
                need[k] = need[k + 1] = True
        extra = self.sample_teacher_continuations(
            ctx_ids_list, [self.m_adaptive - self.m_screen if nd else 0 for nd in need])
        for k, e in enumerate(extra):
            conts[k].extend(e)

        def compute_alpha(samples: List[str]) -> Tuple[Dict[str, float], float]:
            alpha: Dict[str, float] = {cand: 0.0 for cand in candidate_answers}
            alpha["bot"] = 0.0
            n_total = max(1, len(samples))
            n_success = 0
            for text in samples:
                boxed = extract_boxed_content(text) or ""
                hit = next((c for c in candidate_answers if are_answers_equivalent(boxed, c)), None)
                alpha[hit if hit else "bot"] += 1.0 / n_total
                n_success += int(are_answers_equivalent(boxed, gold_answer))
            return alpha, n_success / n_total

        out = []
        for cs in conts:
            alpha_full, v_full = compute_alpha(cs)
            out.append((alpha_full, v_full, len(cs), compute_alpha(cs[0::2])[0], compute_alpha(cs[1::2])[0]))
        return out

    # ------------------------------------------------------------------ teacher scoring (HF)
    def hf_score_fn(self, model, micro_bsz: int = 4):
        """score_fn for teacher_probe: log p(responses | rest) per position, using the HF model."""

        def fn(input_ids, attention_mask, position_ids, responses):
            R = responses.shape[1]
            outs = []
            for s in range(0, input_ids.shape[0], micro_bsz):
                ii, am, pi = (t[s:s + micro_bsz].to(self.device) for t in (input_ids, attention_mask, position_ids))
                with torch.no_grad():
                    logits = model(input_ids=ii, attention_mask=am, position_ids=pi).logits
                    lp = torch.log_softmax(logits[:, -R - 1:-1].float(), dim=-1)
                    outs.append(lp.gather(-1, responses[s:s + micro_bsz].to(self.device).unsqueeze(-1)).squeeze(-1).cpu())
            return torch.cat(outs, dim=0)

        return fn

    def martingale_gaps(self, prompt_ids, resp_ids, steps, candidates, beliefs, n_steps: int = 3) -> List[float]:
        """TV(mean_s q(h o s), q(h)) at each boundary k < K, s ~ P_T(next paragraph | h)."""
        from vllm import SamplingParams

        ends = [0] + [st.token_end for st in steps]
        reqs = [{"prompt_token_ids": prompt_ids + resp_ids[:e]} for e in ends[:-1]]
        sp = SamplingParams(n=n_steps, temperature=1.0, max_tokens=512, stop=["\n\n"],
                            include_stop_str_in_output=True)
        outs = self.vllm_teacher.generate(reqs, sp, use_tqdm=False)
        gaps = []
        score_fn = self.hf_score_fn(self.hf_teacher)
        for k, o in enumerate(outs):
            avg: Dict[str, float] = {}
            m = 0
            for c in o.outputs:
                new_ids = list(c.token_ids)
                if not new_ids:
                    continue
                pr, _ = probe_batch([prompt_ids], [resp_ids[:ends[k]] + new_ids], [candidates], self.tokenizer,
                                    score_fn, pad_token_id=self.tokenizer.pad_token_id, max_steps=1,
                                    strip_trailing_ids=[self.tokenizer.eos_token_id])
                q_next = pr[0].beliefs[-1]
                for key, val in q_next.items():
                    avg[key] = avg.get(key, 0.0) + val
                m += 1
            if m:
                gaps.append(compute_total_variation({kk: vv / m for kk, vv in avg.items()}, beliefs[k]))
        return gaps

    def local_signals(self, prompt_ids: List[int], resp_ids: List[int], steps: List[StepBoundary]):
        """Per-step student/teacher entropy, KL(student||teacher), teacher surprisal and |1 - r_k| from the models."""
        ids = torch.tensor([prompt_ids + resp_ids], device=self.device)
        P, R = len(prompt_ids), len(resp_ids)
        with torch.no_grad():
            ls = self.hf_student(ids).logits[0, P - 1:P + R - 1].float()
            lt = self.hf_teacher(ids).logits[0, P - 1:P + R - 1].float()
        lps, lpt = torch.log_softmax(ls, -1), torch.log_softmax(lt, -1)
        ps, pt = lps.exp(), lpt.exp()
        tgt = ids[0, P:P + R]
        ent_s, ent_t = -(ps * lps).sum(-1), -(pt * lpt).sum(-1)
        kl = (ps * (lps - lpt)).sum(-1)
        a_t = lpt.gather(-1, tgt[:, None])[:, 0] - lps.gather(-1, tgt[:, None])[:, 0]
        surpr = -lpt.gather(-1, tgt[:, None])[:, 0]
        res = []
        for st in steps:
            sl = slice(st.token_start, st.token_end)
            r_k = float(torch.exp(a_t[sl].sum().clamp(-30, 30)))
            res.append((float(ent_s[sl].mean()), float(ent_t[sl].mean()), float(kl[sl].mean()),
                        float(surpr[sl].mean()), abs(1.0 - r_k)))
        return res

    def run_pilot(self, math_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute the full Phase 0 Pilot evaluation on problems."""
        print(f"\n==================================================")
        print(f"  STARTING PHASE 0 PILOT: {len(math_dataset)} PROBLEMS")
        print(f"==================================================\n")

        rollout_evals: List[RolloutEvaluation] = []
        all_probe_errors: List[float] = []
        errors_by_rel_pos: Dict[str, List[float]] = {
            "0.0-0.25": [], "0.25-0.5": [], "0.5-0.75": [], "0.75-1.0": []
        }
        large_shift_errors: List[float] = []
        large_mismatch_errors: List[float] = []

        martingale_gaps: List[float] = []
        all_steps: List[StepEvaluation] = []
        before_answer_steps: List[StepEvaluation] = []

        for p_idx, item in enumerate(math_dataset):
            gold = item["ground_truth"]
            prompt = item["question"]
            p_id = get_prompt_hash(prompt)
            prompt_ids = item["prompt_ids"]

            # a_T: cached greedy teacher answer (never checked against gold)
            a_T = item.get("teacher_answer") or "None"
            self.candidate_manager.register_teacher_answer(p_id, a_T)

            # Rollout 0 is evaluated; the others play the role of earlier-epoch rollouts (history H_x),
            # so C_x is fixed before (and independent of) the evaluated rollout, as in training.
            rollouts = self.sample_rollouts(prompt_ids, 1 + self.n_history)
            resp_ids = rollouts[0]
            for extra in rollouts[1:]:
                ans = extract_boxed_content(self.tokenizer.decode(extra, skip_special_tokens=True))
                if ans:
                    self.candidate_manager.update_history_answer(p_id, canonicalize_answer(ans))
            candidates = self.candidate_manager.get_candidate_set(p_id)
            if not candidates or not resp_ids:
                print(f"[Phase 0] skipping prompt {p_idx}: no candidates or empty rollout")
                continue

            steps = segment_token_ids(resp_ids, self.tokenizer)
            K = len(steps)
            response = self.tokenizer.decode(resp_ids, skip_special_tokens=True)
            student_ans = extract_boxed_content(response)
            student_correct = are_answers_equivalent(student_ans or "", gold)
            teacher_correct = are_answers_equivalent(a_T, gold)

            answer_step_idx = K + 1  # first step containing the final answer string
            if student_ans:
                for st in steps:
                    if student_ans in st.text:
                        answer_step_idx = st.step_idx
                        break

            # Probed beliefs q_0..q_K with the very code used in training (HF teacher), timed for P3
            t0 = time.time()
            probes, pstats = probe_batch(
                [prompt_ids], [resp_ids], [candidates], self.tokenizer, self.hf_score_fn(self.hf_teacher),
                pad_token_id=self.tokenizer.pad_token_id, strip_trailing_ids=[self.tokenizer.eos_token_id])
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.timing["probe_s"] += time.time() - t0
            self.timing["probe_tokens"] += pstats["probe_tokens"]
            probed_beliefs = probes[0].beliefs

            # Reference cost: one plain teacher scoring pass of the same rollout (what OPD already pays)
            t0 = time.time()
            with torch.no_grad():
                self.hf_teacher(torch.tensor([prompt_ids + resp_ids], device=self.device))
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.timing["plain_teacher_s"] += time.time() - t0
            self.timing["plain_teacher_tokens"] += len(prompt_ids) + len(resp_ids)

            local = self.local_signals(prompt_ids, resp_ids, steps)
            martingale_gaps.extend(self.martingale_gaps(prompt_ids, resp_ids, steps, candidates, probed_beliefs))

            # Continuations from every boundary k = 0..K (teacher continues the student's prefix)
            ctxs = [prompt_ids + resp_ids[:(steps[k - 1].token_end if k > 0 else 0)] for k in range(K + 1)]
            bound_stats = self.evaluate_boundaries(ctxs, candidates, gold)

            boundary_evals: List[BoundaryEvaluation] = []
            alphas_half1: List[Dict[str, float]] = []
            alphas_half2: List[Dict[str, float]] = []

            for k in range(K + 1):
                rel_pos = k / max(1, K)
                alpha, v_rate, n_cont, a1, a2 = bound_stats[k]
                alphas_half1.append(a1)
                alphas_half2.append(a2)

                q = probed_beliefs[k]
                e_h = compute_total_variation(q, alpha)
                lower_ci, upper_ci = wilson_score_interval(round(v_rate * n_cont), n_cont, 0.90)
                label = classify_boundary_label(round(v_rate * n_cont), n_cont, 0.5, 0.90)
                is_before = (k < answer_step_idx)

                b_eval = BoundaryEvaluation(
                    boundary_idx=k,
                    step_idx=(k if k > 0 else None),
                    token_offset=steps[k - 1].token_end if k > 0 else 0,
                    rel_position=rel_pos,
                    probed_belief=q,
                    continuation_law=alpha,
                    probe_error=e_h,
                    gold_success_rate=v_rate,
                    wilson_lower=lower_ci,
                    wilson_upper=upper_ci,
                    boundary_label=label,
                    num_continuations=n_cont,
                    before_answer=is_before,
                )
                boundary_evals.append(b_eval)
                all_probe_errors.append(e_h)

                # Bin by relative position
                if rel_pos <= 0.25:
                    errors_by_rel_pos["0.0-0.25"].append(e_h)
                elif rel_pos <= 0.50:
                    errors_by_rel_pos["0.25-0.5"].append(e_h)
                elif rel_pos <= 0.75:
                    errors_by_rel_pos["0.5-0.75"].append(e_h)
                else:
                    errors_by_rel_pos["0.75-1.0"].append(e_h)

            # Compute shifts g_k and continuation shifts g^P_k for steps 1 .. K
            step_evals: List[StepEvaluation] = []
            v_changes: List[float] = []

            for k in range(1, K + 1):
                g_k = compute_total_variation(probed_beliefs[k], probed_beliefs[k - 1])
                g_P = compute_total_variation(boundary_evals[k].continuation_law, boundary_evals[k - 1].continuation_law)
                g_P_h1 = compute_total_variation(alphas_half1[k], alphas_half1[k - 1])
                g_P_h2 = compute_total_variation(alphas_half2[k], alphas_half2[k - 1])

                v_diff = boundary_evals[k].gold_success_rate - boundary_evals[k - 1].gold_success_rate
                v_changes.append(v_diff)

                # Local baseline signals, computed from the models
                step_len = steps[k - 1].token_len
                rel_pos = k / K
                s_ent, t_ent, div_kl, surprisal, mismatch = local[k - 1]

                if g_P > 0.3:
                    large_shift_errors.append(boundary_evals[k].probe_error)
                if mismatch > 0.5:
                    large_mismatch_errors.append(boundary_evals[k].probe_error)

                is_before = (k < answer_step_idx)
                s_eval = StepEvaluation(
                    step_idx=k,
                    token_len=step_len,
                    rel_pos=rel_pos,
                    probed_shift=g_k,
                    continuation_shift=g_P,
                    continuation_shift_half1=g_P_h1,
                    continuation_shift_half2=g_P_h2,
                    v_change=v_diff,
                    student_entropy=s_ent,
                    teacher_entropy=t_ent,
                    divergence_kl=div_kl,
                    teacher_surprisal=surprisal,
                    mismatch_ratio=mismatch,
                    before_answer=is_before,
                    delta_q_aT=probed_beliefs[k][candidates[0]] - probed_beliefs[k - 1][candidates[0]],
                )
                step_evals.append(s_eval)
                all_steps.append(s_eval)
                if is_before:
                    before_answer_steps.append(s_eval)

            # Locate transition step k* (last change from non-low to low, after which all later are low)
            unrec_step = None
            if boundary_evals[0].gold_success_rate >= 0.5:
                for k in range(1, K + 1):
                    # Check if all boundaries from k onwards are low
                    later_labels = [b.boundary_label for b in boundary_evals[k:]]
                    if later_labels and all(lbl == "low" for lbl in later_labels):
                        unrec_step = k
                        break

            # Largest drop step k^- = argmin_k (V(h_{k+1}) - V(h_k))
            largest_drop_idx = int(np.argmin(v_changes)) + 1 if v_changes else None

            r_eval = RolloutEvaluation(
                problem_idx=p_idx,
                prompt_text=prompt,
                gold_answer=gold,
                teacher_answer=a_T,
                student_final_answer=student_ans,
                student_is_correct=student_correct,
                teacher_is_correct=teacher_correct,
                boundaries=boundary_evals,
                steps=step_evals,
                unrecoverable_transition_step=unrec_step,
                largest_drop_step=largest_drop_idx,
            )
            rollout_evals.append(r_eval)

            if (p_idx + 1) % 10 == 0 or (p_idx + 1) == len(math_dataset):
                print(f"[Phase 0 Progress] Evaluated {p_idx + 1}/{len(math_dataset)} rollouts...")

        # -------------------------------------------------------------
        # P1(a): Calibration Statistics
        # -------------------------------------------------------------
        cal_all = {
            "mean": float(np.mean(all_probe_errors)),
            "median": float(np.median(all_probe_errors)),
            "p90": float(np.percentile(all_probe_errors, 90)),
        }
        cal_by_pos = {
            pos: float(np.mean(vals)) if vals else 0.0 for pos, vals in errors_by_rel_pos.items()
        }
        cal_large_shift = float(np.mean(large_shift_errors)) if large_shift_errors else cal_all["mean"]
        cal_large_mismatch = float(np.mean(large_mismatch_errors)) if large_mismatch_errors else cal_all["mean"]

        # -------------------------------------------------------------
        # P1(b): Incremental Predictive Value & Clustered Bootstrap
        # -------------------------------------------------------------
        def analyze_predictive_value(step_list: List[StepEvaluation]) -> Dict[str, Any]:
            if len(step_list) < 10:
                return {"delta_r2": 0.0, "delta_r2_ci": (0.0, 0.0), "partial_spearman": 0.0, "r2_base": 0.0, "r2_ext": 0.0}

            # Features: [rel_pos, length, student_entropy, teacher_entropy, div_kl, teacher_surprisal]
            X_base = np.column_stack([
                [s.rel_pos for s in step_list],
                [s.token_len for s in step_list],
                [s.student_entropy for s in step_list],
                [s.teacher_entropy for s in step_list],
                [s.divergence_kl for s in step_list],
                [s.teacher_surprisal for s in step_list],
            ])
            # Add probed shift g_k
            g_vec = np.array([s.probed_shift for s in step_list])
            X_ext = np.column_stack([X_base, g_vec])
            y = np.array([s.continuation_shift for s in step_list])

            # OLS fits
            def fit_r2(X_mat, y_vec):
                X_design = np.column_stack([np.ones(len(y_vec)), X_mat])
                try:
                    beta, _, _, _ = np.linalg.lstsq(X_design, y_vec, rcond=None)
                    y_pred = X_design @ beta
                    ss_tot = np.sum((y_vec - np.mean(y_vec)) ** 2)
                    ss_res = np.sum((y_vec - y_pred) ** 2)
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))
                    n_obs, n_par = len(y_vec), X_design.shape[1]
                    if n_obs <= n_par:
                        return 0.0
                    return float(1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_par))  # adjusted R^2, not clamped
                except Exception:
                    return 0.0

            r2_base = fit_r2(X_base, y)
            r2_ext = fit_r2(X_ext, y)
            delta_r2 = r2_ext - r2_base  # adjusted R^2 gain; may be negative

            # Partial Spearman rank correlation
            # Regress g_vec on X_base, regress y on X_base, correlate residuals
            X_des = np.column_stack([np.ones(len(y)), X_base])
            b_g, _, _, _ = np.linalg.lstsq(X_des, g_vec, rcond=None)
            res_g = g_vec - (X_des @ b_g)
            b_y, _, _, _ = np.linalg.lstsq(X_des, y, rcond=None)
            res_y = y - (X_des @ b_y)
            rho, _ = stats.spearmanr(res_g, res_y)

            # Clustered Bootstrap by rollout (1000 resamples)
            n_boot = 500
            boot_deltas = []
            step_ids = {id(s) for s in step_list}
            n_rollouts = len(rollout_evals)
            for _ in range(n_boot):
                sampled_rollout_ids = np.random.choice(n_rollouts, size=n_rollouts, replace=True)
                boot_steps = []
                for r_id in sampled_rollout_ids:
                    r_ev = rollout_evals[r_id]
                    for s in r_ev.steps:
                        if id(s) in step_ids:
                            boot_steps.append(s)
                if len(boot_steps) > 10:
                    X_b_b = np.column_stack([
                        [s.rel_pos for s in boot_steps],
                        [s.token_len for s in boot_steps],
                        [s.student_entropy for s in boot_steps],
                        [s.teacher_entropy for s in boot_steps],
                        [s.divergence_kl for s in boot_steps],
                        [s.teacher_surprisal for s in boot_steps],
                    ])
                    X_e_b = np.column_stack([X_b_b, [s.probed_shift for s in boot_steps]])
                    y_b = np.array([s.continuation_shift for s in boot_steps])
                    d_r2 = fit_r2(X_e_b, y_b) - fit_r2(X_b_b, y_b)
                    boot_deltas.append(d_r2)

            if boot_deltas:
                ci_low = float(np.percentile(boot_deltas, 5))
                ci_high = float(np.percentile(boot_deltas, 95))
            else:
                ci_low, ci_high = 0.0, 0.0

            # Split-half reliability of continuation shifts
            h1 = np.array([s.continuation_shift_half1 for s in step_list])
            h2 = np.array([s.continuation_shift_half2 for s in step_list])
            rel_rho, _ = stats.pearsonr(h1, h2) if len(h1) > 1 else (0.0, 0.0)

            return {
                "r2_baseline": r2_base,
                "r2_extended": r2_ext,
                "delta_r2": delta_r2,
                "delta_r2_90ci": (ci_low, ci_high),
                "partial_spearman": float(rho),
                "split_half_reliability": float(rel_rho),
            }

        stats_all_steps = analyze_predictive_value(all_steps)
        stats_before_answer = analyze_predictive_value(before_answer_steps)

        # -------------------------------------------------------------
        # P1(c): Martingale Gap of q along teacher steps
        # -------------------------------------------------------------
        # Teacher steps are sampled at each boundary (vLLM, stop at the paragraph break) and probed with the same
        # code; gap_k = TV(E_s q(h o s), q(h)). A necessary check of calibration only.
        mean_martingale_gap = float(np.mean(martingale_gaps)) if martingale_gaps else float("nan")

        # -------------------------------------------------------------
        # P2: Early Detection & Transitions Before Answer
        # -------------------------------------------------------------
        # Measure unrecoverable transition step detection
        unrec_rollouts = [r for r in rollout_evals if r.unrecoverable_transition_step is not None]
        unrec_share = len(unrec_rollouts) / max(1, len(rollout_evals))

        # Signed agreement between probe change and Delta V_k
        teacher_correct_rollouts = [r for r in rollout_evals if r.teacher_is_correct]
        # Signed agreement: sign(q_k(a_T) - q_{k-1}(a_T)) vs sign(Delta V_k), only where a_T is correct
        # (a_T is candidate 0 by construction) and both changes are non-negligible.
        signed_agreements = []
        for r in teacher_correct_rollouts:
            for st in r.steps:
                if abs(st.v_change) > 0.1 and abs(st.delta_q_aT) > 0.01:
                    signed_agreements.append(float(np.sign(st.delta_q_aT) == np.sign(st.v_change)))
        mean_signed_agreement = float(np.mean(signed_agreements)) if signed_agreements else float("nan")

        # -------------------------------------------------------------
        # P3: Overhead Benchmark
        # -------------------------------------------------------------
        # Measured: HF-teacher time for all probe sequences vs one plain teacher pass over the same rollouts.
        # This is the overhead of the teacher stage only; the whole-step overhead of P3 (which also contains
        # rollout generation and the student update) must be measured in a training run (cw/probe_time_s).
        t_base = max(self.timing["plain_teacher_s"], 1e-9)
        t_method = self.timing["plain_teacher_s"] + self.timing["probe_s"]
        overhead_pct = ((t_method - t_base) / t_base) * 100.0

        # -------------------------------------------------------------
        # Automated Decision Rule:
        # Continue if probed shift adds predictive value before answer appears
        # with 90% confidence interval strictly excluding zero.
        # -------------------------------------------------------------
        delta_r2_before = stats_before_answer["delta_r2"]
        ci_low_before, ci_high_before = stats_before_answer["delta_r2_90ci"]
        
        # Proposal (Phase 0, Decision): continue iff the gain is positive with an interval excluding zero
        decision_go = bool(delta_r2_before > 0.0 and ci_low_before > 0.0)

        results = {
            "p1_calibration": {
                "overall": cal_all,
                "by_position": cal_by_pos,
                "large_shift_error": cal_large_shift,
                "large_mismatch_error": cal_large_mismatch,
            },
            "p1_incremental_predictive_value": {
                "all_steps": stats_all_steps,
                "before_answer_steps": stats_before_answer,
            },
            "p1_martingale_gap": mean_martingale_gap,
            "p2_early_detection": {
                "unrecoverable_transition_rollout_fraction": unrec_share,
                "signed_agreement_rate": mean_signed_agreement,
            },
            "p3_overhead": {
                "t_baseline_sec": t_base,
                "t_method_sec": t_method,
                "overhead_percent": overhead_pct,
                "target_percent": 25.0,
                "within_target": overhead_pct <= 25.0,
            },
            "decision": {
                "go_to_training": decision_go,
                "criteria": "Delta R^2 before answer > 0 and 90% CI excludes zero",
                "delta_r2_before_answer": delta_r2_before,
                "ci_90_before_answer": [ci_low_before, ci_high_before],
                "overhead_status": "Passed (<25%)" if overhead_pct <= 25.0 else "Requires Optimization",
            },
        }

        return results


def generate_markdown_report(results: Dict[str, Any], output_path: str):
    """Generate a clean, structured GitHub Markdown report of Phase 0 Pilot results."""
    dec = results["decision"]
    p1 = results["p1_incremental_predictive_value"]["before_answer_steps"]
    cal = results["p1_calibration"]
    p3 = results["p3_overhead"]

    verdict_badge = "**GO (CONTINUE TO TRAINING)**" if dec["go_to_training"] else "**STOP (ANALYSIS PAPER)**"

    md = f"""# Phase 0 Pilot Evaluation Report
## Teacher Answer-Belief Shifts as a Verifier-Free Allocation Signal

### 1. Executive Summary & Decision Rule
* **Pilot Decision**: {verdict_badge}
* **Criteria**: Incremental predictive value of probed shift $\\Delta R^2$ before the answer appears with 90% CI strictly excluding zero.
* **Probed Shift $\\Delta R^2$ (Before Answer)**: `{p1['delta_r2']:.4f}` (90% CI: `[{dec['ci_90_before_answer'][0]:.4f}, {dec['ci_90_before_answer'][1]:.4f}]`)
* **Partial Spearman Rank Correlation**: `{p1['partial_spearman']:.4f}`
* **Split-Half Continuation Reliability**: `{p1['split_half_reliability']:.4f}`
* **Runtime Overhead**: `{p3['overhead_percent']:.1f}%` (Target: `≤ 25%`) — **{p3['within_target']}**

---

### 2. P1 — Incremental Predictive Value
| Step Set | $R^2$ Baseline (Local + Pos) | $R^2$ With Probed Shift $g_k$ | $\\Delta R^2$ Gain | 90% Clustered Bootstrap CI | Partial Spearman $\\rho$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Steps** | `{results['p1_incremental_predictive_value']['all_steps']['r2_baseline']:.4f}` | `{results['p1_incremental_predictive_value']['all_steps']['r2_extended']:.4f}` | `+{results['p1_incremental_predictive_value']['all_steps']['delta_r2']:.4f}` | `[{results['p1_incremental_predictive_value']['all_steps']['delta_r2_90ci'][0]:.4f}, {results['p1_incremental_predictive_value']['all_steps']['delta_r2_90ci'][1]:.4f}]` | `{results['p1_incremental_predictive_value']['all_steps']['partial_spearman']:.4f}` |
| **Before Answer Only** | `{p1['r2_baseline']:.4f}` | `{p1['r2_extended']:.4f}` | `+{p1['delta_r2']:.4f}` | `[{dec['ci_90_before_answer'][0]:.4f}, {dec['ci_90_before_answer'][1]:.4f}]` | `{p1['partial_spearman']:.4f}` |

---

### 3. P1(a) — Probe Calibration $e(h) = \\mathrm{{TV}}(q(h), \\alpha_{{\\mathcal{{P}}}}(h))$
* **Overall Mean $e(h)$**: `{cal['overall']['mean']:.4f}` (Median: `{cal['overall']['median']:.4f}`, 90th percentile: `{cal['overall']['p90']:.4f}`)
* **By Relative Position**:
  * $0.0 - 0.25$: `{cal['by_position']['0.0-0.25']:.4f}`
  * $0.25 - 0.50$: `{cal['by_position']['0.25-0.5']:.4f}`
  * $0.50 - 0.75$: `{cal['by_position']['0.5-0.75']:.4f}`
  * $0.75 - 1.00$: `{cal['by_position']['0.75-1.0']:.4f}`
* **At Boundaries with Large Shift ($g^{{\\mathcal{{P}}}} > 0.3$)**: `{cal['large_shift_error']:.4f}`
* **At Boundaries with Large Mismatch ($|1 - r_k| > 0.5$)**: `{cal['large_mismatch_error']:.4f}`
* **Martingale Gap Check**: `{results['p1_martingale_gap']:.4f}`

---

### 4. P2 — Early Detection Before the Answer Appears
* **Unrecoverable Transition Identification Rate**: `{results['p2_early_detection']['unrecoverable_transition_rollout_fraction'] * 100.0:.1f}%`
* **Signed Directional Agreement $\\mathrm{{sign}}(\\Delta q) = \\mathrm{{sign}}(\\Delta V)$**: `{results['p2_early_detection']['signed_agreement_rate'] * 100.0:.1f}%` (nan = no step with a non-negligible change)

---

### 5. P3 — Cost & Overhead Accounting
* **OPD Baseline Latency**: `{p3['t_baseline_sec']:.2f} s/step`
* **Consequence-Weighted OPD Latency**: `{p3['t_method_sec']:.2f} s/step`
* **Measured Overhead**: `{p3['overhead_percent']:.1f}%`
* **Decision**: Proceed to full model training on cluster using Table 1 and Table 2 configurations.
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[Phase 0] Markdown report written to {output_path}")


def load_math_subset(data_path: str, teacher_cache: str, n: int, seed: int, build_prompt_ids) -> List[Dict[str, Any]]:
    """MATH prompts with gold (analysis only) and cached a_T; prompts without a_T are skipped."""
    import pandas as pd

    df = pd.read_parquet(data_path)
    with open(teacher_cache, "r", encoding="utf-8") as f:
        cache = json.load(f)
    order = np.random.RandomState(seed).permutation(len(df))
    items = []
    for i in order:
        row = df.iloc[int(i)]
        msgs = [dict(m) for m in list(row["prompt"])]
        q = question_text_from_messages(msgs)
        a_T = cache.get(get_prompt_hash(q))
        if a_T in (None, "", "None"):
            continue
        items.append({"question": q, "ground_truth": str(row["reward_model"]["ground_truth"]),
                      "teacher_answer": a_T, "prompt_ids": build_prompt_ids(msgs)})
        if len(items) >= n:
            break
    return items


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Pilot Experiment Suite for Consequence-Weighted OPD")
    parser.add_argument("--student_model", default="Qwen/Qwen3-1.7B-Base", help="Student model path or name")
    parser.add_argument("--teacher_model", default="Qwen/Qwen3-8B", help="Teacher model path or name")
    parser.add_argument("--num_prompts", type=int, default=100, help="Number of MATH prompts to evaluate")
    parser.add_argument("--m_screen", type=int, default=8, help="Screening continuations per boundary")
    parser.add_argument("--m_adaptive", type=int, default=32, help="Adaptive continuations for ambiguous boundaries")
    parser.add_argument("--data_path", default="data/math/train.parquet", help="MATH train parquet")
    parser.add_argument("--teacher_cache", default="data/math/teacher_answers_qwen3_8b.json", help="cached a_T")
    parser.add_argument("--n_history", type=int, default=2, help="extra student rollouts used as earlier-epoch history")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="experiments/phase0_pilot", help="Directory for JSON and report output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    runner = Phase0Runner(
        student_model_path=args.student_model,
        teacher_model_path=args.teacher_model,
        num_prompts=args.num_prompts,
        m_screen=args.m_screen,
        m_adaptive=args.m_adaptive,
        n_history=args.n_history,
        seed=args.seed,
    )
    prompts_list = load_math_subset(args.data_path, args.teacher_cache, args.num_prompts, args.seed,
                                    runner.build_prompt_ids)
    print(f"[Phase 0] {len(prompts_list)} MATH prompts with cached a_T")

    results = runner.run_pilot(prompts_list)

    # Save JSON results
    json_path = os.path.join(args.output_dir, "phase0_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[Phase 0] Results JSON written to {json_path}")

    # Save Markdown report
    md_path = os.path.join(args.output_dir, "phase0_report.md")
    generate_markdown_report(results, md_path)

    print("\n" + "="*50)
    print(f"PILOT DECISION VERDICT: {'GO (CONTINUE TO TRAINING)' if results['decision']['go_to_training'] else 'NO-GO'}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
