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
Token-exact teacher probing for CW-OPD (Section 3.3 of proposal v7).

Everything here works on *token ids*, so the context the teacher sees at a step
boundary is exactly the prefix of the tokens the student sampled (no decode /
re-encode round trip, no inserted separators):

    context_k = prompt_ids ++ response_ids[:end_k] ++ probe_ids        (k = 0..K)
    q_k(a)    = prod_j p_T(tau_j | context_k, tau_<j),  a ∘ "}" = tau_1..tau_m

The module is independent of Ray / FSDP: the teacher is injected as a callable
``score_fn(input_ids, attention_mask, position_ids, responses) -> log_probs``
that returns the per-position log-probability of ``responses`` (B, R). The
trainer supplies a callable backed by the teacher worker group; unit tests and
Phase 0 supply CPU / HF callables.

Cost note: without KV-cache sharing or a tree attention mask every (boundary,
candidate) pair is an independent sequence, i.e. roughly (K+1)*|C_x| extra
teacher sequences per rollout. ``max_steps`` caps K to bound this.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from verl.trainer.ppo.cw_opd.probe_scorer import PROBE_STRING, StepBoundary

BOT_KEY = "bot"

# score_fn(input_ids, attention_mask, position_ids, responses) -> (B, R) log-probs of `responses`
ScoreFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


# --------------------------------------------------------------------------------------
# Step segmentation on token ids
# --------------------------------------------------------------------------------------
class TokenTextCache:
    """Caches ``decode([token_id])`` (single-token decoding) for a tokenizer."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self._cache: Dict[int, str] = {}

    def __call__(self, token_id: int) -> str:
        s = self._cache.get(token_id)
        if s is None:
            s = self.tokenizer.decode([token_id], skip_special_tokens=False)
            self._cache[token_id] = s
        return s


def _break_positions(resp_ids: Sequence[int], text_of: Callable[[int], str], separator: str) -> List[int]:
    """Exclusive end indices of paragraphs (a paragraph ends after the token that completes ``separator``)."""
    ends: List[int] = []
    tail = ""  # last len(separator)-1 characters seen
    keep = max(0, len(separator) - 1)
    has_content = False  # current paragraph has non-whitespace text
    for i, tid in enumerate(resp_ids):
        s = text_of(tid)
        combined = tail + s
        if separator in combined and (has_content or s.strip()):
            ends.append(i + 1)
            has_content = False
            tail = ""
            continue
        if s.strip():
            has_content = True
        tail = combined[-keep:] if keep else ""
    return ends


def _cap_steps(ends: List[int], total: int, max_steps: int) -> List[int]:
    """Reduce the number of steps to ``max_steps`` by keeping the boundaries closest to even token spacing."""
    if max_steps <= 0 or len(ends) <= max_steps:
        return ends
    inner = [e for e in ends if e < total]
    targets = [total * (j + 1) / max_steps for j in range(max_steps - 1)]
    chosen: List[int] = []
    for t in targets:
        best = min(inner, key=lambda e: abs(e - t))
        if best not in chosen:
            chosen.append(best)
    chosen.sort()
    return chosen + [total]


def segment_token_ids(
    resp_ids: Sequence[int],
    tokenizer: Any,
    separator: str = "\n\n",
    max_steps: int = 0,
    text_of: Optional[Callable[[int], str]] = None,
) -> List[StepBoundary]:
    """Split a response (valid tokens only) into paragraph steps with exact token boundaries.

    A step ends after the token in which ``separator`` is completed. If a response has no
    paragraph break at all, single newlines are used. The last step always ends at len(resp_ids).
    """
    total = len(resp_ids)
    if total == 0:
        return [StepBoundary(step_idx=1, token_start=0, token_end=0, token_len=0, text="")]
    text_of = text_of or TokenTextCache(tokenizer)

    ends = _break_positions(resp_ids, text_of, separator)
    if not ends and separator != "\n":
        ends = _break_positions(resp_ids, text_of, "\n")
    if not ends or ends[-1] != total:
        ends = ends + [total]
    ends = _cap_steps(ends, total, max_steps)

    steps: List[StepBoundary] = []
    start = 0
    for idx, end in enumerate(ends):
        text = tokenizer.decode(list(resp_ids[start:end]), skip_special_tokens=False)
        steps.append(
            StepBoundary(step_idx=idx + 1, token_start=start, token_end=end, token_len=end - start, text=text)
        )
        start = end
    return steps


# --------------------------------------------------------------------------------------
# Probe tokenization and request construction
# --------------------------------------------------------------------------------------
def tokenize_probe_and_candidate(
    tokenizer: Any, candidate: str, probe_string: str = PROBE_STRING
) -> Tuple[List[int], List[int], bool]:
    """Return (probe_ids, cand_ids, retokenized) for the string ``probe + candidate + "}"``.

    The text is tokenized jointly and split at the probe boundary when the probe tokens are a
    prefix of the joint tokenization (the natural tokenization of ``\\boxed{<a>}``). Otherwise the
    two parts are tokenized separately and ``retokenized`` is True (such mass lands in the residual cell).
    """
    probe_ids = tokenizer.encode(probe_string, add_special_tokens=False)
    full = tokenizer.encode(probe_string + candidate + "}", add_special_tokens=False)
    if full[: len(probe_ids)] == probe_ids and len(full) > len(probe_ids):
        return probe_ids, full[len(probe_ids):], False
    return probe_ids, tokenizer.encode(candidate + "}", add_special_tokens=False), True


@dataclass
class ProbeRequest:
    rollout: int
    boundary: int  # k in 0..K (probe after the first k steps)
    cand_idx: int
    context_ids: List[int]
    cand_ids: List[int]


def build_probe_requests(
    rollout: int,
    prompt_ids: Sequence[int],
    resp_ids: Sequence[int],
    steps: Sequence[StepBoundary],
    candidates: Sequence[str],
    probe_ids: Sequence[int],
    cand_ids_list: Sequence[Sequence[int]],
    max_context_tokens: Optional[int] = None,
) -> List[ProbeRequest]:
    """All (boundary, candidate) requests for one rollout; ``steps`` must tile ``resp_ids``.

    ``max_context_tokens`` truncates the response part of the context (used to drop a trailing EOS token
    so the probe is appended to the text, not after the end-of-sequence marker).
    """
    cap = len(resp_ids) if max_context_tokens is None else max_context_tokens
    ends = [0] + [min(s.token_end, cap) for s in steps]
    reqs: List[ProbeRequest] = []
    for k, end in enumerate(ends):
        ctx = list(prompt_ids) + list(resp_ids[:end]) + list(probe_ids)
        for c, cand_ids in enumerate(cand_ids_list):
            reqs.append(ProbeRequest(rollout, k, c, ctx, list(cand_ids)))
    return reqs


def pack_requests(
    requests: Sequence[ProbeRequest], pad_token_id: int
) -> Dict[str, torch.Tensor]:
    """Left-pad contexts, right-pad candidates; returns tensors in verl's (prompt | response) layout."""
    n = len(requests)
    max_ctx = max(len(r.context_ids) for r in requests)
    max_cand = max(len(r.cand_ids) for r in requests)
    seq = max_ctx + max_cand
    input_ids = torch.full((n, seq), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((n, seq), dtype=torch.long)
    responses = torch.full((n, max_cand), pad_token_id, dtype=torch.long)
    for i, r in enumerate(requests):
        lc, la = len(r.context_ids), len(r.cand_ids)
        input_ids[i, max_ctx - lc: max_ctx] = torch.tensor(r.context_ids, dtype=torch.long)
        input_ids[i, max_ctx: max_ctx + la] = torch.tensor(r.cand_ids, dtype=torch.long)
        attention_mask[i, max_ctx - lc: max_ctx + la] = 1
        responses[i, :la] = torch.tensor(r.cand_ids, dtype=torch.long)
    position_ids = (torch.cumsum(attention_mask, dim=-1) - 1).clamp(min=0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
    }


def balanced_order(lengths: Sequence[int], dp_size: int) -> List[int]:
    """Permutation that sorts by length and deals items round-robin so contiguous chunks of size n/dp are balanced."""
    n = len(lengths)
    if dp_size <= 1 or n == 0:
        return list(range(n))
    order = sorted(range(n), key=lambda i: -lengths[i])
    buckets: List[List[int]] = [[] for _ in range(dp_size)]
    for j, idx in enumerate(order):
        buckets[j % dp_size].append(idx)
    return [i for b in buckets for i in b]


def score_requests(
    requests: Sequence[ProbeRequest],
    score_fn: ScoreFn,
    pad_token_id: int,
    chunk_size: int = 256,
    dp_size: int = 1,
) -> List[float]:
    """log q for each request: sum of teacher log-probs over the candidate tokens (incl. closing brace)."""
    n = len(requests)
    if n == 0:
        return []
    lengths = [len(r.context_ids) + len(r.cand_ids) for r in requests]
    perm = balanced_order(lengths, dp_size)
    out = [0.0] * n
    chunk_size = max(1, chunk_size)
    if dp_size > 1:
        chunk_size = max(dp_size, (chunk_size // dp_size) * dp_size)
    for s in range(0, n, chunk_size):
        idxs = perm[s: s + chunk_size]
        reqs = [requests[i] for i in idxs]
        packed = pack_requests(reqs, pad_token_id)
        lp = score_fn(packed["input_ids"], packed["attention_mask"], packed["position_ids"], packed["responses"])
        lp = lp.detach().float().cpu()
        for row, (i, r) in enumerate(zip(idxs, reqs)):
            out[i] = float(lp[row, : len(r.cand_ids)].sum().item())
    return out


# --------------------------------------------------------------------------------------
# Beliefs
# --------------------------------------------------------------------------------------
@dataclass
class RolloutProbe:
    """Probe result for one rollout."""

    steps: List[StepBoundary]
    candidates: List[str]
    beliefs: List[Dict[str, float]]  # q_0..q_K, keyed by candidate string and BOT_KEY
    overflow: int = 0  # number of boundaries with sum_a q(a) > 1 (numerical)
    retokenized: int = 0  # number of candidates whose probe/answer split was not the natural tokenization
    extra: Dict[str, Any] = field(default_factory=dict)


def assemble_belief(cands: Sequence[str], log_qs: Sequence[float]) -> Tuple[Dict[str, float], bool]:
    """q(a)=exp(log q), q(bot)=max(0, 1-sum). Not renormalized (proposal Eq. 3). Returns (q, overflowed)."""
    q: Dict[str, float] = {}
    total = 0.0
    for a, lq in zip(cands, log_qs):
        p = math.exp(min(lq, 0.0))  # a log-prob cannot exceed 0; clamp numerical noise
        q[a] = p
        total += p
    overflow = total > 1.0 + 1e-6
    q[BOT_KEY] = max(0.0, 1.0 - total)
    return q, overflow


def probe_batch(
    prompt_ids_list: Sequence[Sequence[int]],
    resp_ids_list: Sequence[Sequence[int]],
    candidates_list: Sequence[Sequence[str]],
    tokenizer: Any,
    score_fn: ScoreFn,
    pad_token_id: int,
    probe_string: str = PROBE_STRING,
    max_steps: int = 0,
    chunk_size: int = 256,
    dp_size: int = 1,
    separator: str = "\n\n",
    strip_trailing_ids: Sequence[int] = (),
) -> Tuple[List[RolloutProbe], Dict[str, float]]:
    """Segment every rollout, probe the teacher at all K+1 boundaries and return beliefs + cost stats."""
    text_of = TokenTextCache(tokenizer)
    strip_set = set(strip_trailing_ids)
    cand_cache: Dict[str, Tuple[List[int], List[int], bool]] = {}

    all_requests: List[ProbeRequest] = []
    per_rollout: List[Tuple[List[StepBoundary], List[str], List[bool]]] = []
    for i, (p_ids, r_ids, cands) in enumerate(zip(prompt_ids_list, resp_ids_list, candidates_list)):
        steps = segment_token_ids(r_ids, tokenizer, separator=separator, max_steps=max_steps, text_of=text_of)
        cands = list(cands)
        retok_flags: List[bool] = []
        cand_ids_list: List[List[int]] = []
        probe_ids: List[int] = []
        for a in cands:
            if a not in cand_cache:
                cand_cache[a] = tokenize_probe_and_candidate(tokenizer, a, probe_string)
            probe_ids, c_ids, retok = cand_cache[a]
            cand_ids_list.append(c_ids)
            retok_flags.append(retok)
        if not cands:
            probe_ids = tokenizer.encode(probe_string, add_special_tokens=False)
        ctx_len = len(r_ids)
        while ctx_len > 0 and r_ids[ctx_len - 1] in strip_set:
            ctx_len -= 1
        if cands:
            all_requests.extend(
                build_probe_requests(i, p_ids, r_ids, steps, cands, probe_ids, cand_ids_list, ctx_len)
            )
        per_rollout.append((steps, cands, retok_flags))

    log_qs = score_requests(all_requests, score_fn, pad_token_id, chunk_size=chunk_size, dp_size=dp_size)

    # gather: (rollout, boundary) -> [log q per candidate]
    table: Dict[Tuple[int, int], List[float]] = {}
    for req, lq in zip(all_requests, log_qs):
        table.setdefault((req.rollout, req.boundary), [0.0] * len(per_rollout[req.rollout][1]))[req.cand_idx] = lq

    probes: List[RolloutProbe] = []
    for i, (steps, cands, retok_flags) in enumerate(per_rollout):
        beliefs: List[Dict[str, float]] = []
        overflow = 0
        for k in range(len(steps) + 1):
            if cands:
                q, ov = assemble_belief(cands, table[(i, k)])
                overflow += int(ov)
            else:
                q = {BOT_KEY: 1.0}  # no candidate known: all mass in the residual cell, g_k = 0
            beliefs.append(q)
        probes.append(
            RolloutProbe(
                steps=steps,
                candidates=cands,
                beliefs=beliefs,
                overflow=overflow,
                retokenized=int(sum(retok_flags)),
            )
        )

    stats = {
        "probe_sequences": float(len(all_requests)),
        "probe_tokens": float(sum(len(r.context_ids) + len(r.cand_ids) for r in all_requests)),
    }
    return probes, stats
