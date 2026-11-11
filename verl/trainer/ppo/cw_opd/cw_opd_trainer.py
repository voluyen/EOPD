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
Consequence-Weighted On-Policy Distillation Trainer (CWOPDTrainer).

Extends OnPolicyDistillTrainer with (Algorithm 1 of proposal v7):
- candidate sets C_x = {a_T} U H_x fixed before the rollout (CandidateManager)
- token-exact paragraph segmentation of the student rollout
- *real* teacher probing at all K+1 boundaries through the teacher worker group
  (the reference-policy slot), giving q_0..q_K and g_k = TV(q_k, q_{k-1})
- batch-normalised stop-gradient token weights written to ``batch["cw_weights"]``
- update of H_x with the rollout's own answer *after* weights are computed

Everything that is computed is also logged: scalar metrics go to the normal trainer
logger (console / wandb / ...) under ``cw/``, and per-rollout probe traces are written as
JSONL under ``cw_opd.log_dir`` every ``cw_opd.dump_every`` steps.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.cw_opd.candidate_manager import (
    CandidateManager,
    are_answers_equivalent,
    canonicalize_answer,
    extract_boxed_content,
    extract_question_from_chat_prompt,
    get_prompt_hash,
)
from verl.trainer.ppo.cw_opd.probe_scorer import (
    compute_consequence_weights,
    shifts_from_beliefs,
)
from verl.trainer.ppo.cw_opd.teacher_probe import BOT_KEY, RolloutProbe, probe_batch
from verl.trainer.ppo.on_policy_distill_trainer import OnPolicyDistillTrainer


def _pct(values: List[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else 0.0


class CWOPDTrainer(OnPolicyDistillTrainer):
    """Consequence-Weighted On-Policy Distillation Trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        cw_config = self.config.get("cw_opd", {}) or {}
        teacher_cache_path = cw_config.get("teacher_cache_path", None)
        if not teacher_cache_path:
            teacher_cache_path = self.config.actor_rollout_ref.get("teacher_model", {}).get("cache_path", None)

        self.candidate_manager = CandidateManager(
            teacher_cache_path=teacher_cache_path,
            max_candidates=cw_config.get("max_candidates", 3),
        )
        if teacher_cache_path and not self.candidate_manager.teacher_cache:
            raise RuntimeError(
                f"cw_opd.teacher_cache_path={teacher_cache_path} is missing or empty: without a_T every candidate "
                "set is empty and all weights collapse to 1 (plain OPD). Run scripts/run_cache_teacher.sh first."
            )

        # "none" | "at_only" | "length_norm" | "kl_variant" | "permutation"
        self.ablation_mode = cw_config.get("ablation_mode", "none")
        self.max_steps = int(cw_config.get("max_steps", 0))  # 0 = no cap on K
        self.probe_chunk_size = int(cw_config.get("probe_chunk_size", 256))
        self.dump_every = int(cw_config.get("dump_every", 10))  # 0 disables JSONL traces
        self.seed = int(cw_config.get("seed", 0))
        log_dir = cw_config.get("log_dir", None)
        if not log_dir:
            log_dir = os.path.join(self.config.trainer.get("default_local_dir", "."), "cw_opd_logs")
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.history_path = os.path.join(self.log_dir, "answer_history.json")
        self.candidate_manager.load_history(self.history_path)

        self._extra_metrics: Dict[str, float] = {}
        self._cum_probe_tokens = 0.0
        self._cum_probe_time = 0.0
        print(
            f"[CWOPDTrainer] ablation_mode={self.ablation_mode} max_steps={self.max_steps} "
            f"max_candidates={self.candidate_manager.max_candidates} log_dir={self.log_dir} "
            f"teacher_answers={len(self.candidate_manager.teacher_cache)}"
        )

    # ------------------------------------------------------------------ teacher access
    def _teacher_score_fn(self, input_ids, attention_mask, position_ids, responses) -> torch.Tensor:
        """Teacher log-prob of ``responses`` given the rest of each sequence, via the teacher worker group."""
        if getattr(self, "use_legacy_worker_impl", "auto") == "disable":
            raise RuntimeError("CW-OPD probing needs the legacy worker path (use_legacy_worker_impl != 'disable').")
        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
            }
        )
        data, pad_size = pad_dataproto_to_divisor(data, self.ref_policy_wg.world_size)
        out = self.ref_policy_wg.compute_ref_log_prob(data)
        out = unpad_dataproto(out, pad_size=pad_size)
        return out.batch["ref_log_prob"]

    # ------------------------------------------------------------------ helpers
    def _question_keys(self, batch: DataProto, prompts: torch.Tensor, prompt_mask: torch.Tensor) -> List[str]:
        keys = []
        for i in range(prompts.shape[0]):
            ids = prompts[i][prompt_mask[i].bool()].tolist()
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            keys.append(get_prompt_hash(extract_question_from_chat_prompt(text)))
        return keys

    def _special_tail_ids(self) -> List[int]:
        ids = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id, 151643, 151645}
        return [i for i in ids if i is not None]

    # ------------------------------------------------------------------ main computation
    def _compute_consequence_weights(self, batch: DataProto):
        responses = batch.batch["responses"]
        bsz, resp_len = responses.shape
        prompts = batch.batch["prompts"]
        attention_mask = batch.batch["attention_mask"]
        prompt_len = prompts.shape[1]
        prompt_mask = attention_mask[:, :prompt_len]
        if "response_mask" in batch.batch.keys():
            resp_mask = batch.batch["response_mask"]
        else:
            resp_mask = attention_mask[:, prompt_len:]
        device = responses.device

        keys = self._question_keys(batch, prompts, prompt_mask)
        prompt_ids = [prompts[i][prompt_mask[i].bool()].tolist() for i in range(bsz)]
        resp_ids = [responses[i][resp_mask[i].bool()].tolist() for i in range(bsz)]

        # 1) candidate sets, fixed before this rollout is used
        at_only = self.ablation_mode == "at_only"
        cands = [self.candidate_manager.get_candidate_set(k, at_only=at_only) for k in keys]
        cache_miss = [self.candidate_manager.get_teacher_answer(k) is None for k in keys]

        # 2) probe the teacher at every boundary
        t0 = time.time()
        probes, probe_stats = probe_batch(
            prompt_ids_list=prompt_ids,
            resp_ids_list=resp_ids,
            candidates_list=cands,
            tokenizer=self.tokenizer,
            score_fn=self._teacher_score_fn,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0,
            max_steps=self.max_steps,
            chunk_size=self.probe_chunk_size,
            dp_size=self.ref_policy_wg.world_size,
            strip_trailing_ids=self._special_tail_ids(),
        )
        probe_time = time.time() - t0

        # 3) shifts (variant applied) and plain TV (always, for logging)
        batch_shifts: List[List[float]] = []
        batch_tv: List[List[float]] = []
        batch_steps = []
        for i, pr in enumerate(probes):
            seed = self.seed * 1_000_003 + self.global_steps * 1009 + i
            batch_shifts.append(shifts_from_beliefs(pr.beliefs, pr.steps, self.ablation_mode, seed=seed))
            batch_tv.append(shifts_from_beliefs(pr.beliefs, pr.steps, "none"))
            batch_steps.append(pr.steps)

        # 4) weights
        w_tilde, stats = compute_consequence_weights(
            batch_shifts=batch_shifts,
            batch_steps=batch_steps,
            response_length=resp_len,
            batch_size=bsz,
            response_mask=resp_mask,
            device=device,
        )

        # 5) answers (analysis / history only) and history update AFTER weights
        student_answers = []
        for i in range(bsz):
            text = self.tokenizer.decode(resp_ids[i], skip_special_tokens=True)
            boxed = extract_boxed_content(text)
            student_answers.append(canonicalize_answer(boxed) if boxed else None)
        for k, text_ans in zip(keys, student_answers):
            if text_ans:
                self.candidate_manager.update_history_answer(k, text_ans)

        metrics = self._summarize(
            probes=probes,
            batch_tv=batch_tv,
            batch_shifts=batch_shifts,
            w_tilde=w_tilde,
            resp_mask=resp_mask,
            resp_ids=resp_ids,
            cands=cands,
            cache_miss=cache_miss,
            student_answers=student_answers,
            stats=stats,
            probe_stats=probe_stats,
            probe_time=probe_time,
        )

        if self.dump_every > 0 and self.global_steps % self.dump_every == 0:
            self._dump_traces(batch, keys, probes, batch_tv, batch_shifts, w_tilde, cands, student_answers)
            self.candidate_manager.save_history(self.history_path)

        return w_tilde, metrics

    # ------------------------------------------------------------------ logging
    def _summarize(
        self,
        probes: List[RolloutProbe],
        batch_tv,
        batch_shifts,
        w_tilde,
        resp_mask,
        resp_ids,
        cands,
        cache_miss,
        student_answers,
        stats,
        probe_stats,
        probe_time,
    ) -> Dict[str, float]:
        m: Dict[str, float] = {}
        bsz = len(probes)
        flat_tv = [g for gs in batch_tv for g in gs]
        flat_shift = [g for gs in batch_shifts for g in gs]
        K = [len(p.steps) for p in probes]
        step_len = [s.token_len for p in probes for s in p.steps]

        # shift distribution (plain TV, comparable across ablations)
        m["cw/g_tv_mean_step"] = float(np.mean(flat_tv)) if flat_tv else 0.0
        for q in (50, 90, 99):
            m[f"cw/g_tv_p{q}"] = _pct(flat_tv, q)
        m["cw/g_tv_max"] = float(np.max(flat_tv)) if flat_tv else 0.0
        m["cw/frac_steps_g_gt_0.1"] = float(np.mean(np.array(flat_tv) > 0.1)) if flat_tv else 0.0
        m["cw/frac_steps_g_gt_0.3"] = float(np.mean(np.array(flat_tv) > 0.3)) if flat_tv else 0.0
        m["cw/frac_rollouts_all_zero_shift"] = float(np.mean([max(g, default=0.0) < 1e-8 for g in batch_tv]))
        m["cw/g_used_mean_step"] = float(np.mean(flat_shift)) if flat_shift else 0.0  # after variant (e.g. g/L)
        m["cw/g_bar_B"] = stats["cw_shift_mean"]

        # where the largest shift occurs (position confound check)
        rel = [
            (int(np.argmax(g)) + 1) / len(g) for g in batch_tv if len(g) > 0 and max(g) > 1e-8
        ]
        m["cw/argmax_shift_rel_pos_mean"] = float(np.mean(rel)) if rel else 0.0

        # weights
        valid = resp_mask.bool()
        w_valid = w_tilde[valid]
        m["cw/w_mean"] = float(w_valid.mean().item()) if w_valid.numel() else 1.0
        m["cw/w_max"] = float(w_valid.max().item()) if w_valid.numel() else 1.0
        m["cw/w_min"] = float(w_valid.min().item()) if w_valid.numel() else 1.0
        m["cw/w_std"] = float(w_valid.std().item()) if w_valid.numel() > 1 else 0.0
        m["cw/w_top10pct_token_share"] = stats["cw_weight_concentration"]
        total_w = float(w_valid.sum().item()) if w_valid.numel() else 1.0
        max_step_share = 0.0
        for i, pr in enumerate(probes):
            for s in pr.steps:
                lo, hi = s.token_start, min(s.token_end, w_tilde.shape[1])
                if hi > lo:
                    max_step_share = max(max_step_share, float(w_tilde[i, lo:hi].sum().item()) / max(total_w, 1e-8))
        m["cw/w_max_single_step_share"] = max_step_share

        # steps / candidates
        m["cw/K_mean"] = float(np.mean(K)) if K else 0.0
        m["cw/K_max"] = float(np.max(K)) if K else 0.0
        m["cw/step_len_mean"] = float(np.mean(step_len)) if step_len else 0.0
        m["cw/step_len_max"] = float(np.max(step_len)) if step_len else 0.0
        n_c = [len(c) for c in cands]
        m["cw/n_candidates_mean"] = float(np.mean(n_c)) if n_c else 0.0
        m["cw/frac_prompts_only_aT"] = float(np.mean([n <= 1 for n in n_c])) if n_c else 0.0
        m["cw/frac_prompts_no_candidates"] = float(np.mean([n == 0 for n in n_c])) if n_c else 0.0
        m["cw/aT_cache_miss_frac"] = float(np.mean(cache_miss)) if cache_miss else 0.0
        m["cw/history_prompts"] = float(len(self.candidate_manager.history))

        # beliefs
        q0_aT, qK_aT, q0_bot, qK_bot, q0_sum, qK_sum = [], [], [], [], [], []
        for pr, c, miss in zip(probes, cands, cache_miss):
            if not c or miss:  # c[0] is a_T only when it is cached
                continue
            first, last = pr.beliefs[0], pr.beliefs[-1]
            q0_aT.append(first[c[0]])
            qK_aT.append(last[c[0]])
            q0_bot.append(first[BOT_KEY])
            qK_bot.append(last[BOT_KEY])
            q0_sum.append(1.0 - first[BOT_KEY])
            qK_sum.append(1.0 - last[BOT_KEY])
        for name, vals in (
            ("q0_aT", q0_aT), ("qK_aT", qK_aT), ("q0_bot", q0_bot), ("qK_bot", qK_bot),
            ("q0_cand_mass", q0_sum), ("qK_cand_mass", qK_sum),
        ):
            m[f"cw/{name}"] = float(np.mean(vals)) if vals else 0.0
        m["cw/q_overflow_boundaries"] = float(sum(p.overflow for p in probes))
        n_cand_total = max(1, sum(len(c) for c in cands))
        m["cw/frac_candidates_retokenized"] = float(sum(p.retokenized for p in probes)) / n_cand_total

        # agreement of the rollout's own answer with a_T (no gold involved)
        agree, noans = [], []
        for ans, c, miss in zip(student_answers, cands, cache_miss):
            noans.append(ans is None)
            if ans is not None and c and not miss:
                agree.append(are_answers_equivalent(ans, c[0]))
        m["cw/student_answer_missing_frac"] = float(np.mean(noans)) if noans else 0.0
        m["cw/student_agrees_with_aT_frac"] = float(np.mean(agree)) if agree else 0.0

        # cost
        resp_tokens = float(sum(len(r) for r in resp_ids))
        m["cw/probe_sequences"] = probe_stats["probe_sequences"]
        m["cw/probe_tokens"] = probe_stats["probe_tokens"]
        m["cw/probe_tokens_per_response_token"] = probe_stats["probe_tokens"] / max(resp_tokens, 1.0)
        m["cw/probe_time_s"] = probe_time
        self._cum_probe_tokens += probe_stats["probe_tokens"]
        self._cum_probe_time += probe_time
        m["cw/cum_probe_tokens"] = self._cum_probe_tokens
        m["cw/cum_probe_time_s"] = self._cum_probe_time

        self._extra_metrics = m
        print(
            f"[CW-OPD step {self.global_steps}] K={m['cw/K_mean']:.1f} |C|={m['cw/n_candidates_mean']:.2f} "
            f"g_tv(mean/p90/max)={m['cw/g_tv_mean_step']:.3f}/{m['cw/g_tv_p90']:.3f}/{m['cw/g_tv_max']:.3f} "
            f"w_max={m['cw/w_max']:.2f} maxStepShare={m['cw/w_max_single_step_share']:.3f} "
            f"q0_aT={m['cw/q0_aT']:.3f} qK_aT={m['cw/qK_aT']:.3f} probe={probe_time:.1f}s "
            f"({m['cw/probe_tokens_per_response_token']:.1f}x resp tokens) cache_miss={m['cw/aT_cache_miss_frac']:.2f}"
        )
        if m["cw/frac_prompts_no_candidates"] > 0.05:
            print(
                f"[CW-OPD WARNING] {m['cw/frac_prompts_no_candidates']:.0%} of prompts have no candidate answers "
                "(a_T cache miss?) -> their weights are 1."
            )
        return m

    def _dump_traces(self, batch, keys, probes, batch_tv, batch_shifts, w_tilde, cands, student_answers):
        """JSONL: one line per rollout with beliefs, shifts and weights per step (for offline analysis)."""
        gold = None
        try:  # analysis only; never used in training
            rm = batch.non_tensor_batch.get("reward_model", None)
            if rm is not None:
                gold = [r.get("ground_truth") if isinstance(r, dict) else None for r in rm]
        except Exception:
            gold = None
        path = os.path.join(self.log_dir, f"probe_step{self.global_steps:06d}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, pr in enumerate(probes):
                w_steps = [
                    float(w_tilde[i, s.token_start: s.token_end].mean().item()) if s.token_end > s.token_start else 1.0
                    for s in pr.steps
                ]
                rec = {
                    "step": self.global_steps,
                    "prompt_key": keys[i],
                    "candidates": cands[i],
                    "student_answer": student_answers[i],
                    "gold_analysis_only": gold[i] if gold is not None else None,
                    "steps": [{"start": s.token_start, "end": s.token_end, "len": s.token_len} for s in pr.steps],
                    "beliefs": pr.beliefs,
                    "g_tv": batch_tv[i],
                    "g_used": batch_shifts[i],
                    "w_step": w_steps,
                    "retokenized": pr.retokenized,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _consume_extra_metrics(self) -> Dict[str, float]:
        m, self._extra_metrics = self._extra_metrics, {}
        return m

    # ------------------------------------------------------------------ hook into the trainer step
    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Compute consequence weights (real teacher probing), then the usual teacher log-probs."""
        w_tilde, _ = self._compute_consequence_weights(batch)
        batch.batch["cw_weights"] = w_tilde.float()
        return super()._compute_ref_log_prob(batch)
