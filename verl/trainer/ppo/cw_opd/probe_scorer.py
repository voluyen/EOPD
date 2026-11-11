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
Probe Scorer and Shift Allocator for Consequence-Weighted OPD (CW-OPD).

Implements Section 3.3, 3.4, and 3.5 of proposal_consequence_weighted_opd_v7:
- Paragraph segmentation into reasoning steps s_1, ..., s_K
- Probe string π = "Therefore, the final answer is \\boxed{"
- Teacher probability calculation q_k(a) for candidates a in C_x
- Residual cell q_k(⊥) = 1 - sum_{a in C_x} q_k(a)
- Total variation shift g_k = TV(q_k, q_{k-1})
- Batch token-weighted mean g_bar_B and stop-gradient weights w_k, w_tilde_t
- Support for Table 1 / Table 2 ablations:
    * at_only: C_x = {a_T}
    * permutation: stratified shuffle within relative position and length
    * length_norm: g_k / L_k allocation
    * kl_variant: KL(q_k || q_{k-1})
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


PROBE_STRING = "Therefore, the final answer is \\boxed{"


@dataclass
class StepBoundary:
    """Represents a reasoning step / paragraph boundary."""
    step_idx: int          # 1-indexed: 1 .. K
    token_start: int       # Start token index in response (inclusive)
    token_end: int         # End token index in response (exclusive)
    token_len: int         # Number of tokens L_k
    text: str              # Step text s_k


def split_response_into_steps(
    response_text: str,
    response_tokens: Optional[List[int]] = None,
    tokenizer: Optional[Any] = None,
    min_tokens_per_step: int = 16,
) -> List[StepBoundary]:
    """Split a student response into reasoning steps / paragraphs.
    
    Paragraph breaks are primarily detected by double newlines '\\n\\n'.
    If the response has no double newlines, falls back to single newline '\\n'.
    Aligns character boundaries with token boundaries.
    """
    if not response_text or not response_text.strip():
        # Fallback for empty response
        resp_len = len(response_tokens) if response_tokens else 1
        return [StepBoundary(step_idx=1, token_start=0, token_end=resp_len, token_len=resp_len, text="")]

    # Split by double newline or single newline
    paragraphs = [p for p in re.split(r"\n\s*\n", response_text) if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [p for p in response_text.split("\n") if p.strip()]
    if not paragraphs:
        paragraphs = [response_text]

    # If tokenizer and tokens are provided, compute exact token slice alignment
    if tokenizer is not None and response_tokens is not None and len(response_tokens) > 0:
        total_tokens = len(response_tokens)
        steps: List[StepBoundary] = []
        curr_tok = 0
        
        # Compute incremental token lengths
        for idx, para in enumerate(paragraphs):
            is_last = (idx == len(paragraphs) - 1)
            para_toks = tokenizer.encode(para, add_special_tokens=False)
            para_len = max(1, len(para_toks))
            
            start_tok = curr_tok
            if is_last:
                end_tok = total_tokens
            else:
                end_tok = min(total_tokens, curr_tok + para_len)
                
            step_len = max(1, end_tok - start_tok)
            steps.append(
                StepBoundary(
                    step_idx=idx + 1,
                    token_start=start_tok,
                    token_end=end_tok,
                    token_len=step_len,
                    text=para,
                )
            )
            curr_tok = end_tok
            if curr_tok >= total_tokens:
                break
                
        # Merge tiny trailing steps if necessary
        if len(steps) > 1 and steps[-1].token_len < min_tokens_per_step:
            last = steps.pop()
            steps[-1].token_end = last.token_end
            steps[-1].token_len += last.token_len
            steps[-1].text += "\n" + last.text
            
        return steps
    else:
        # Approximate token boundary by character ratio if tokens not provided
        total_chars = max(1, len(response_text))
        approx_total_toks = len(response_tokens) if response_tokens is not None else max(1, total_chars // 4)
        
        steps = []
        curr_tok = 0
        for idx, para in enumerate(paragraphs):
            is_last = (idx == len(paragraphs) - 1)
            para_tok_count = max(1, round(len(para) / total_chars * approx_total_toks))
            start_tok = curr_tok
            end_tok = approx_total_toks if is_last else min(approx_total_toks, curr_tok + para_tok_count)
            steps.append(
                StepBoundary(
                    step_idx=idx + 1,
                    token_start=start_tok,
                    token_end=end_tok,
                    token_len=max(1, end_tok - start_tok),
                    text=para,
                )
            )
            curr_tok = end_tok
        return steps


def compute_total_variation(q_curr: Dict[str, float], q_prev: Dict[str, float]) -> float:
    """Compute Total Variation distance TV(q_curr, q_prev) between two cell distributions.
    
    Formula (Equation 4 & 5):
    TV(mu, nu) = 0.5 * ( sum_{a in C_x} |mu(a) - nu(a)| + |mu(bot) - nu(bot)| )
    """
    all_keys = set(q_curr.keys()).union(set(q_prev.keys()))
    diff_sum = 0.0
    for k in all_keys:
        val_curr = q_curr.get(k, 0.0)
        val_prev = q_prev.get(k, 0.0)
        diff_sum += abs(val_curr - val_prev)
    tv = 0.5 * diff_sum
    return float(np.clip(tv, 0.0, 1.0))


def compute_kl_divergence(q_curr: Dict[str, float], q_prev: Dict[str, float], eps: float = 1e-12) -> float:
    """Compute KL divergence KL(q_curr || q_prev) for Ablation T2(c)."""
    all_keys = set(q_curr.keys()).union(set(q_prev.keys()))
    kl = 0.0
    for k in all_keys:
        p = max(eps, q_curr.get(k, eps))
        q = max(eps, q_prev.get(k, eps))
        kl += p * math.log(p / q)
    return max(0.0, float(kl))


def permute_shifts_stratified(
    shifts: List[float],
    steps: List[StepBoundary],
    n_pos_bins: int = 3,
    n_len_bins: int = 2,
    seed: Optional[int] = None,
) -> List[float]:
    """Permutation control for Table 1 (T1(6)).
    
    Permutes g_k among steps within strata of relative position (k/K) and length (L_k),
    preserving the exact weight distribution.
    """
    if len(shifts) <= 1:
        return list(shifts)

    rng = np.random.RandomState(seed)
    K = len(steps)
    
    # Stratify by (pos_bin, len_bin)
    lengths = [s.token_len for s in steps]
    len_median = float(np.median(lengths)) if lengths else 1.0

    strata: Dict[Tuple[int, int], List[int]] = {}
    for i, s in enumerate(steps):
        rel_pos = (i + 1) / K
        pos_bin = min(n_pos_bins - 1, int(rel_pos * n_pos_bins))
        len_bin = 1 if s.token_len >= len_median else 0
        key = (pos_bin, len_bin)
        if key not in strata:
            strata[key] = []
        strata[key].append(i)

    permuted = list(shifts)
    for key, indices in strata.items():
        if len(indices) > 1:
            vals = [shifts[idx] for idx in indices]
            rng.shuffle(vals)
            for idx, val in zip(indices, vals):
                permuted[idx] = val

    return permuted


def shifts_from_beliefs(
    beliefs: List[Dict[str, float]],
    steps: List[StepBoundary],
    ablation_mode: str = "none",
    seed: Optional[int] = None,
) -> List[float]:
    """g_k = TV(q_k, q_{k-1}) for k = 1..K from beliefs q_0..q_K, with the T1/T2 variants applied."""
    K = len(steps)
    assert len(beliefs) == K + 1, f"need K+1={K + 1} beliefs, got {len(beliefs)}"
    shifts: List[float] = []
    for k in range(1, K + 1):
        if ablation_mode == "kl_variant":
            g = compute_kl_divergence(beliefs[k], beliefs[k - 1])
        else:
            g = compute_total_variation(beliefs[k], beliefs[k - 1])
        if ablation_mode == "length_norm":
            g = g / max(1, steps[k - 1].token_len)
        shifts.append(g)
    if ablation_mode == "permutation":
        shifts = permute_shifts_stratified(shifts, steps, seed=seed)
    return shifts


class ProbeScorer:
    """Computes probed answer beliefs q_k and shifts g_k using the teacher model."""

    def __init__(
        self,
        probe_string: str = PROBE_STRING,
        ablation_mode: str = "none",  # "none", "at_only", "length_norm", "kl_variant", "permutation"
    ):
        self.probe_string = probe_string
        self.ablation_mode = ablation_mode

    def score_candidates_batch(
        self,
        teacher_model: Any,
        tokenizer: Any,
        prefix_texts: List[str],
        candidate_answers: List[str],
        device: torch.device = torch.device("cpu"),
        allow_mock: bool = False,
    ) -> List[Dict[str, float]]:
        """Score candidate answers following probe prefix given a teacher model (text-based, HF model).

        NOTE: training uses the token-exact path in ``teacher_probe.py``; this text-based scorer is kept
        for small HF-model use. ``teacher_model=None`` is only allowed with ``allow_mock=True`` (unit tests).
        
        For each prefix:
            appends probe string π
            scores each candidate followed by "}" -> a o "}"
            computes q_k(a) = prod_{j=1}^m p_T(tau_j | prefix o π o tau_{<j})
            computes q_k(⊥) = max(0, 1 - sum_{a} q_k(a))
            
        Returns:
            List of dictionaries, each mapping candidate string and "bot" to probabilities.
        """
        results: List[Dict[str, float]] = []

        # If model is None (e.g. testing or mock), generate dummy uniform/mock distribution
        if teacher_model is None:
            if not allow_mock:
                raise ValueError(
                    "teacher_model is None: refusing to return a dummy belief (it would silently give g_k = 0). "
                    "Pass allow_mock=True only in unit tests."
                )
            for _ in prefix_texts:
                dist = {}
                n = len(candidate_answers)
                if n > 0:
                    for a in candidate_answers:
                        dist[a] = 0.7 / n
                    dist["bot"] = 0.3
                else:
                    dist["bot"] = 1.0
                results.append(dist)
            return results

        teacher_model.eval()

        for prefix in prefix_texts:
            full_probe_prefix = prefix + self.probe_string
            prefix_ids = tokenizer.encode(full_probe_prefix, add_special_tokens=False)
            
            cand_probs: Dict[str, float] = {}
            total_cand_prob = 0.0

            for cand in candidate_answers:
                # String to score: a o "}"
                cand_str = f"{cand}}}"
                cand_ids = tokenizer.encode(cand_str, add_special_tokens=False)
                if not cand_ids:
                    cand_probs[cand] = 0.0
                    continue

                full_input_ids = torch.tensor([prefix_ids + cand_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = teacher_model(full_input_ids).logits  # [1, seq_len, vocab_size]
                    log_probs = F.log_softmax(logits, dim=-1)

                    # Gather log probs for candidate tokens
                    # Prefix length: len(prefix_ids)
                    # Candidate token tau_j is predicted by token at index len(prefix_ids) - 1 + j
                    cand_len = len(cand_ids)
                    log_prob_sum = 0.0
                    for j in range(cand_len):
                        pred_pos = len(prefix_ids) - 1 + j
                        tok_id = cand_ids[j]
                        log_prob_sum += log_probs[0, pred_pos, tok_id].item()

                prob = math.exp(log_prob_sum)
                cand_probs[cand] = prob
                total_cand_prob += prob

            # Residual cell q(bot) = max(0, 1 - sum_a q(a)); candidates are NOT renormalized
            cand_probs["bot"] = max(0.0, 1.0 - total_cand_prob)

            results.append(cand_probs)

        return results

    def compute_rollout_shifts(
        self,
        prompt_text: str,
        response_text: str,
        candidate_answers: List[str],
        teacher_model: Any,
        tokenizer: Any,
        response_tokens: Optional[List[int]] = None,
        device: torch.device = torch.device("cpu"),
        seed: Optional[int] = None,
        allow_mock: bool = False,
    ) -> Tuple[List[StepBoundary], List[float], List[Dict[str, float]]]:
        """Compute reasoning step shifts g_k for a single student rollout.
        
        Args:
            prompt_text: Problem prompt x.
            response_text: Student rollout response y.
            candidate_answers: Candidate set C_x (fixed before rollout).
            teacher_model: Teacher model for scoring.
            tokenizer: Tokenizer.
            response_tokens: Optional list of token ids for response.
            device: PyTorch device.
            seed: Random seed for permutation control.
            
        Returns:
            steps: List of StepBoundary (s_1 .. s_K).
            shifts: List of shifts g_k (k = 1 .. K).
            beliefs: List of belief distributions q_k (k = 0 .. K).
        """
        steps = split_response_into_steps(
            response_text=response_text,
            response_tokens=response_tokens,
            tokenizer=tokenizer,
        )
        K = len(steps)

        # Prepare prefixes h_1 .. h_{K+1}
        # k=0: prompt alone (h_1)
        # k=1..K: prompt + reasoning up to step k (h_{k+1})
        prefixes: List[str] = [prompt_text]
        curr_text = prompt_text
        for s in steps:
            curr_text = curr_text + "\n\n" + s.text if curr_text else s.text
            prefixes.append(curr_text)

        # If ablation mode is at_only, restrict candidate_answers to first candidate (a_T)
        cands = candidate_answers
        if self.ablation_mode == "at_only" and len(cands) > 1:
            cands = cands[:1]

        # Score all prefixes k = 0 .. K
        beliefs = self.score_candidates_batch(
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            prefix_texts=prefixes,
            candidate_answers=cands,
            device=device,
            allow_mock=allow_mock,
        )

        shifts = shifts_from_beliefs(beliefs, steps, self.ablation_mode, seed=seed)

        return steps, shifts, beliefs


def compute_consequence_weights(
    batch_shifts: List[List[float]],
    batch_steps: List[List[StepBoundary]],
    response_length: int,
    batch_size: int,
    response_mask: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute normalized token weights w_tilde_t across a rollout batch.
    
    Formula (Equation 6):
        w_k = 1 + g_k / g_bar_B
        g_bar_B = (1/N) * sum_{t in B} g_{k(t)}
        w_tilde_t = (N * w_{k(t)}) / (sum_{t' in B} w_{k(t')})
        
    Args:
        batch_shifts: List of shifts [g_1 .. g_K] per rollout sequence in batch.
        batch_steps: List of StepBoundary lists per sequence in batch.
        response_length: Maximum response length tensor dimension.
        batch_size: Batch size.
        response_mask: Optional mask tensor of shape (batch_size, response_length).
        device: Target tensor device.
        
    Returns:
        w_tilde: Tensor of shape (batch_size, response_length) with batch mean 1.0.
        stats: Dictionary of diagnostic metrics (g_bar_B, max_weight, Gini concentration).
    """
    token_shifts = torch.zeros((batch_size, response_length), dtype=torch.float32, device=device)

    for i in range(batch_size):
        if i >= len(batch_shifts) or i >= len(batch_steps):
            continue
        shifts = batch_shifts[i]
        steps = batch_steps[i]
        
        for k, step in enumerate(steps):
            if k < len(shifts):
                g_val = float(shifts[k])
                start = max(0, min(response_length, step.token_start))
                end = max(0, min(response_length, step.token_end))
                if end > start:
                    token_shifts[i, start:end] = g_val

    # Apply response mask to consider valid tokens only
    if response_mask is None:
        mask = (token_shifts > -1e-5).float()
    else:
        mask = response_mask.to(device).float()

    num_valid_tokens = mask.sum().item()
    if num_valid_tokens == 0:
        return torch.ones((batch_size, response_length), dtype=torch.float32, device=device), {
            "cw_shift_mean": 0.0,
            "cw_weight_max": 1.0,
            "cw_weight_concentration": 0.0,
        }

    # Batch token-weighted mean shift g_bar_B
    g_bar_B = (token_shifts * mask).sum().item() / (num_valid_tokens + 1e-8)

    # Step weights w_k = 1 + g_k / g_bar_B
    if g_bar_B > 1e-8:
        w_raw = 1.0 + (token_shifts / g_bar_B)
    else:
        w_raw = torch.ones_like(token_shifts)

    # Normalize token weights so mean over valid tokens in batch is exactly 1.0
    w_sum = (w_raw * mask).sum().item()
    if w_sum > 1e-8:
        w_tilde = (num_valid_tokens * w_raw * mask) / w_sum
    else:
        w_tilde = torch.ones_like(token_shifts) * mask

    # Diagnostics: Weight concentration & top share
    valid_weights = w_tilde[mask.bool()].detach().cpu().numpy()
    max_weight = float(np.max(valid_weights)) if len(valid_weights) > 0 else 1.0
    
    # Top 10% weight share
    if len(valid_weights) > 10:
        k_top = max(1, len(valid_weights) // 10)
        sorted_w = np.sort(valid_weights)[::-1]
        top_10_share = float(np.sum(sorted_w[:k_top]) / np.sum(sorted_w))
    else:
        top_10_share = 0.10

    stats = {
        "cw_shift_mean": float(g_bar_B),
        "cw_weight_max": max_weight,
        "cw_weight_concentration": top_10_share,
    }

    return w_tilde, stats
