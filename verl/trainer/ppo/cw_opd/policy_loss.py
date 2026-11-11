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
Policy Loss Functions for Consequence-Weighted OPD (CW-OPD), CW-EOPD, and FUW.

Implements Section 3.4 of proposal_consequence_weighted_opd_v7:
- cw_opd: Clipped PPO surrogate with advantage w_tilde_t * a_t
- cw_eopd: CW-OPD combined with EOPD entropy-gated forward-KL Soft KD,
           where w_tilde_t also multiplies the per-token forward-KL term.
- fuw: Token utility trust-region reweighting baseline (fuw2026).
"""

from typing import Any, Dict, Optional, Tuple

import torch
import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("cw_opd")
def compute_policy_loss_cw_opd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    ref_log_prob: Optional[torch.Tensor] = None,
    cw_weights: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer: Any = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute Consequence-Weighted On-Policy Distillation (CW-OPD) loss.
    
    Formula:
        a_t = (teacher_log_prob - old_log_prob).detach()
        adv_t = w_tilde_t * a_t
        loss = -max(adv_t * ratio, adv_t * clamp(ratio, 1 - eps_low, 1 + eps_high))
    """
    assert config is not None, "config is required for compute_policy_loss_cw_opd"
    if ref_log_prob is None:
        raise ValueError("ref_log_prob (teacher) is required for cw_opd")

    teacher_log_prob = ref_log_prob

    # PPO ratio = exp(log_prob - old_log_prob)
    old_log_prob = old_log_prob.detach()
    ratio = torch.exp(log_prob - old_log_prob)

    # Base teacher advantage a_t = (teacher_log_prob - old_log_prob).detach()
    teacher_advantage = (teacher_log_prob - old_log_prob).detach()

    # Determine consequence weights w_tilde_t
    # Check cw_weights or rollout_is_weights
    w_tilde = cw_weights if cw_weights is not None else rollout_is_weights
    if w_tilde is not None:
        assert w_tilde.shape == teacher_advantage.shape, (
            f"cw_weights shape {w_tilde.shape} mismatch with teacher_advantage {teacher_advantage.shape}"
        )
        final_advantage = (w_tilde.to(teacher_advantage.device) * teacher_advantage).detach()
    else:
        final_advantage = teacher_advantage

    clip_ratio_low = getattr(config, "clip_ratio_low", config.clip_ratio)
    clip_ratio_high = getattr(config, "clip_ratio_high", config.clip_ratio)

    # Clipped PPO surrogate
    pg_losses1 = -final_advantage * ratio
    pg_losses2 = -final_advantage * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Aggregate token loss
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # Metrics
    kl_divergence = verl_F.masked_mean(-teacher_advantage, response_mask)
    avg_teacher_advantage = verl_F.masked_mean(teacher_advantage, response_mask)
    avg_weighted_advantage = verl_F.masked_mean(final_advantage, response_mask)
    pg_clipfrac = (ratio < (1 - clip_ratio_low)).float() + (ratio > (1 + clip_ratio_high)).float()
    pg_clipfrac = verl_F.masked_mean(pg_clipfrac, response_mask)
    ppo_kl = verl_F.masked_mean(log_prob - old_log_prob, response_mask)

    metrics = {
        "actor/rev_kl": kl_divergence.detach().item(),
        "actor/avg_teacher_advantage": avg_teacher_advantage.detach().item(),
        "actor/avg_weighted_advantage": avg_weighted_advantage.detach().item(),
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }

    if w_tilde is not None:
        metrics["actor/cw_weight_mean"] = verl_F.masked_mean(w_tilde, response_mask).detach().item()

    return pg_loss, metrics


@register_policy_loss("cw_eopd")
def compute_policy_loss_cw_eopd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    ref_log_prob: Optional[torch.Tensor] = None,
    cw_weights: Optional[torch.Tensor] = None,
    teacher_entropy: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    tokenizer: Any = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute Consequence-Weighted OPD combined with EOPD.
    
    Formula:
        Both the clipped PPO surrogate and the EOPD forward-KL soft KD loss
        are reweighted by w_tilde_t (stop-gradient).
    """
    assert config is not None
    if ref_log_prob is None:
        raise ValueError("ref_log_prob is required for cw_eopd")

    # Step 1: Compute CW-OPD policy gradient loss
    pg_loss, metrics = compute_policy_loss_cw_opd(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
        ref_log_prob=ref_log_prob,
        cw_weights=cw_weights,
        teacher_entropy=teacher_entropy,
        teacher_topk_log_probs=teacher_topk_log_probs,
        student_topk_log_probs=student_topk_log_probs,
        input_ids=input_ids,
        tokenizer=tokenizer,
        **kwargs,
    )

    # Step 2: Compute Soft KD loss reweighted by w_tilde_t
    soft_kd_loss = torch.tensor(0.0, device=log_prob.device)
    if teacher_topk_log_probs is not None and student_topk_log_probs is not None:
        if teacher_entropy is None:
            raise RuntimeError("teacher_entropy is required for cw_eopd")

        teacher_probs = torch.exp(teacher_topk_log_probs)
        # Forward KL per token
        kl_per_token = torch.sum(teacher_probs * (teacher_topk_log_probs - student_topk_log_probs), dim=-1)

        soft_kd_coef = getattr(config, "soft_kd_loss_coef", 1.0)
        soft_kd_threshold = getattr(config, "soft_kd_entropy_threshold", 0.8)

        soft_kd_entropy_mask = (teacher_entropy >= soft_kd_threshold).float()
        soft_kd_mask = response_mask * soft_kd_entropy_mask

        # Weight the forward KL term by w_tilde_t
        w_tilde = cw_weights if cw_weights is not None else rollout_is_weights
        if w_tilde is not None:
            kl_per_token = kl_per_token * w_tilde.to(kl_per_token.device)

        total_valid_tokens = response_mask.sum()
        soft_kd_loss = (
            verl_F.masked_sum(kl_per_token, soft_kd_mask) / (total_valid_tokens + 1e-8)
        ) * soft_kd_coef

        metrics["actor/cw_soft_kd_loss"] = soft_kd_loss.detach().item()

    final_loss = pg_loss + soft_kd_loss
    return final_loss, metrics


@register_policy_loss("fuw")
def compute_policy_loss_fuw(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: Optional[torch.Tensor] = None,
    ref_log_prob: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Token Utility for On-Policy Distillation (FUW baseline, fuw2026).
    
    Weights token advantages by local trust-region utility u_t derived from |a_t|.
    """
    assert config is not None
    if ref_log_prob is None:
        raise ValueError("ref_log_prob is required for fuw")

    old_log_prob = old_log_prob.detach()
    ratio = torch.exp(log_prob - old_log_prob)
    teacher_advantage = (ref_log_prob - old_log_prob).detach()

    # Token utility: u_t = clamp(|a_t|, 0.1, 2.0)
    utility_raw = torch.clamp(torch.abs(teacher_advantage), min=0.1, max=2.0)
    
    # Normalize utility over batch tokens
    valid_tokens = response_mask.sum().item()
    if valid_tokens > 0:
        u_mean = (utility_raw * response_mask).sum().item() / valid_tokens
        u_norm = utility_raw / (u_mean + 1e-8)
    else:
        u_norm = utility_raw

    final_advantage = (u_norm * teacher_advantage).detach()

    clip_ratio_low = getattr(config, "clip_ratio_low", config.clip_ratio)
    clip_ratio_high = getattr(config, "clip_ratio_high", config.clip_ratio)

    pg_losses1 = -final_advantage * ratio
    pg_losses2 = -final_advantage * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    metrics = {
        "actor/rev_kl": verl_F.masked_mean(-teacher_advantage, response_mask).detach().item(),
        "actor/fuw_advantage": verl_F.masked_mean(final_advantage, response_mask).detach().item(),
        "actor/pg_clipfrac": verl_F.masked_mean(
            (ratio < (1 - clip_ratio_low)).float() + (ratio > (1 + clip_ratio_high)).float(),
            response_mask,
        ).detach().item(),
    }

    return pg_loss, metrics
