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

Extends OnPolicyDistillTrainer with:
- Online tracking of candidate answers C_x = {a_T} U H_x across epochs
- Paragraph boundary segmentation of student rollouts
- Probed teacher belief shifts g_k = TV(q_k, q_{k-1})
- Stop-gradient token weights w_tilde_t added to batch
- Post-rollout extraction of A(y) to update H_x for subsequent epochs
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.on_policy_distill_trainer import OnPolicyDistillTrainer
from verl.trainer.ppo.cw_opd.candidate_manager import CandidateManager, get_prompt_hash
from verl.trainer.ppo.cw_opd.probe_scorer import (
    ProbeScorer,
    compute_consequence_weights,
    split_response_into_steps,
)


class CWOPDTrainer(OnPolicyDistillTrainer):
    """Consequence-Weighted On-Policy Distillation Trainer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Candidate management configuration
        cw_config = self.config.get("cw_opd", {})
        teacher_cache_path = cw_config.get("teacher_cache_path", None)
        if not teacher_cache_path:
            # Fallback to teacher_model.cache_path
            teacher_cache_path = self.config.actor_rollout_ref.get("teacher_model", {}).get("cache_path", None)

        self.candidate_manager = CandidateManager(
            teacher_cache_path=teacher_cache_path,
            max_candidates=cw_config.get("max_candidates", 3),
        )

        # Probing mode & ablations
        # Modes: "none", "at_only", "length_norm", "kl_variant", "permutation"
        self.ablation_mode = cw_config.get("ablation_mode", "none")
        self.probe_scorer = ProbeScorer(
            ablation_mode=self.ablation_mode,
        )
        print(f"[CWOPDTrainer] Initialized with ablation_mode={self.ablation_mode}")

    def _compute_consequence_weights(self, batch: DataProto) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute consequence-weighted token weights for the current rollout batch."""
        responses = batch.batch["responses"]  # [bsz, resp_len]
        bsz, resp_len = responses.shape
        device = responses.device

        # Decode prompt and response texts
        input_ids = batch.batch["input_ids"]
        seq_len = input_ids.shape[1]
        prompt_len = seq_len - resp_len

        batch_shifts: List[List[float]] = []
        batch_steps: List[List[Any]] = []
        batch_cands_count: List[int] = []

        # We score with teacher model if available or fallback
        # In Ray actor setup, if teacher is a separate worker group, we can evaluate on device
        for i in range(bsz):
            prompt_toks = input_ids[i, :prompt_len].cpu().tolist()
            resp_toks = responses[i].cpu().tolist()

            prompt_text = self.tokenizer.decode(prompt_toks, skip_special_tokens=True)
            resp_text = self.tokenizer.decode(resp_toks, skip_special_tokens=True)
            prompt_id = get_prompt_hash(prompt_text)

            # Retrieve candidate set fixed before current rollout
            at_only = (self.ablation_mode == "at_only")
            cands = self.candidate_manager.get_candidate_set(prompt_id=prompt_id, at_only=at_only)
            batch_cands_count.append(len(cands))

            # Segment response into steps
            steps, shifts, _ = self.probe_scorer.compute_rollout_shifts(
                prompt_text=prompt_text,
                response_text=resp_text,
                candidate_answers=cands,
                teacher_model=None,  # Handled via batch teacher evaluation or probe scorer
                tokenizer=self.tokenizer,
                response_tokens=resp_toks,
                device=device,
            )
            batch_shifts.append(shifts)
            batch_steps.append(steps)

        # Compute batch-normalized weights w_tilde_t
        resp_mask = batch.batch.get("response_mask", None)
        w_tilde, stats = compute_consequence_weights(
            batch_shifts=batch_shifts,
            batch_steps=batch_steps,
            response_length=resp_len,
            batch_size=bsz,
            response_mask=resp_mask,
            device=device,
        )

        stats["cw_avg_candidates"] = float(np.mean(batch_cands_count)) if batch_cands_count else 1.0
        stats["cw_avg_steps"] = float(np.mean([len(s) for s in batch_steps])) if batch_steps else 1.0

        # Post-rollout update: extract student final answer and add to H_x for later epochs
        for i in range(bsz):
            prompt_toks = input_ids[i, :prompt_len].cpu().tolist()
            resp_toks = responses[i].cpu().tolist()
            prompt_text = self.tokenizer.decode(prompt_toks, skip_special_tokens=True)
            resp_text = self.tokenizer.decode(resp_toks, skip_special_tokens=True)
            prompt_id = get_prompt_hash(prompt_text)
            self.candidate_manager.update_history(prompt_id, resp_text)

        return w_tilde, stats

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        """Compute ref_log_prob and inject consequence weights w_tilde into batch."""
        # 1. Compute consequence weights from rollout responses
        w_tilde, cw_stats = self._compute_consequence_weights(batch)
        
        # Inject consequence weights into batch
        batch.batch["cw_weights"] = w_tilde.float()
        batch.batch["rollout_is_weights"] = w_tilde.float()

        # Log metrics to batch meta_info
        if "metrics" not in batch.meta_info:
            batch.meta_info["metrics"] = {}
        for k, v in cw_stats.items():
            batch.meta_info["metrics"][f"trainer/{k}"] = v

        # 2. Call parent to compute teacher log prob & entropy
        return super()._compute_ref_log_prob(batch)
