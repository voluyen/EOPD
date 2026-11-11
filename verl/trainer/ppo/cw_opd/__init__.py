# Copyright 2026 The Consequence-Weighted OPD Authors.

from verl.trainer.ppo.cw_opd.candidate_manager import CandidateManager, extract_boxed_content, canonicalize_answer
from verl.trainer.ppo.cw_opd.probe_scorer import (
    ProbeScorer,
    StepBoundary,
    split_response_into_steps,
    compute_total_variation,
    compute_kl_divergence,
    compute_consequence_weights,
)
from verl.trainer.ppo.cw_opd.policy_loss import (
    compute_policy_loss_cw_opd,
    compute_policy_loss_cw_eopd,
    compute_policy_loss_fuw,
)

__all__ = [
    "CandidateManager",
    "extract_boxed_content",
    "canonicalize_answer",
    "ProbeScorer",
    "StepBoundary",
    "split_response_into_steps",
    "compute_total_variation",
    "compute_kl_divergence",
    "compute_consequence_weights",
    "compute_policy_loss_cw_opd",
    "compute_policy_loss_cw_eopd",
    "compute_policy_loss_fuw",
    "CWOPDTrainer",
]


def __getattr__(name):
    """Lazy import: the trainer pulls in ray / the full PPO stack, which the pure-python parts do not need."""
    if name == "CWOPDTrainer":
        from verl.trainer.ppo.cw_opd.cw_opd_trainer import CWOPDTrainer

        return CWOPDTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
