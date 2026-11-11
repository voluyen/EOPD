# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A fork of **verl** (ByteDance's Ray-based RL post-training library) used as the base for a research project: **Consequence-Weighted On-Policy Distillation (CW-OPD)**, i.e. "Teacher Answer-Belief Shifts as a Verifier-Free Allocation Signal for On-Policy Distillation". Nearly everything under `verl/` is upstream; the project-specific code is small and listed below. `CW_OPD_GUIDE.md` (in Vietnamese) is the experiment runbook. The proposal `.tex`/`.pdf` is gitignored (`proposal*`, `*.tex`, `*.pdf`), so it may or may not be present locally.

Setup: Qwen3-8B (non-thinking) teacher → Qwen3-1.7B-Base student, trained on MATH. Evaluated on MATH500, AIME24/25, AMC23, Minerva, and OlympiadBench using the Qwen2.5-Math protocol (8 samples, Avg@8/Pass@8, paired bootstrap).

## Commands

```bash
pip install -r requirements.txt            # plus requirements-test.txt for tests

# CW-OPD unit tests (CPU-only; the GPU path is never exercised)
pytest tests/test_cw_opd.py -q
pytest tests/test_cw_opd.py::test_balanced_braces -q   # single test

# Upstream verl tests: files named *_on_cpu.py run on CPU; everything else assumes GPUs
pytest tests/test_protocol_on_cpu.py -q

# Lint / format (ruff, line-length 120; mypy; license-header + docstring checks)
pre-commit run --all-files
```

Experiment pipeline (needs CUDA GPUs; the defaults are `N_GPUS=4`, `NNODES=1`, overridable through env vars like `STUDENT_MODEL`, `TEACHER_MODEL`, `TEACHER_CACHE`, `CKPTS_DIR`):

```bash
bash scripts/run_all_experiments_master.sh {setup|models|check|data|cache|phase0|table1|table2|eval|all}
bash scripts/run_train_t1_<variant>.sh      # opd, eopd, fuw, ours_cw_opd, ours_cw_eopd, permutation
bash scripts/run_train_t2_ablation_<x>.sh   # at_only, length_norm, kl
bash scripts/run_eval_all_benchmarks.sh <checkpoint_dir>
```

Data goes under `data/math/` and `data/eval_benchmarks/` (parquet files are gitignored). The teacher answer cache is `data/math/teacher_answers_qwen3_8b.json`, and each training script regenerates it if it is missing.

## Architecture

### How a distillation run is wired

`verl/trainer/main_ppo.py` (Hydra entry, config `verl/trainer/config/ppo_trainer.yaml`) picks the trainer class from `trainer.trainer_class`:
- `RayPPOTrainer` (default, `verl/trainer/ppo/ray_trainer.py`): the upstream PPO loop.
- `OnPolicyDistillTrainer` (`verl/trainer/ppo/on_policy_distill_trainer.py`): this fork's OPD/EOPD baseline. It reuses the **reference-policy slot as the teacher**. `actor_rollout_ref.teacher_model.path` is loaded into `Role.RefPolicy`, so `ref_log_prob`, `ref_entropy`, and `ref_topk_log_probs/indices` in the batch are *teacher* quantities. It also swaps `<|endoftext|>` for `<|im_end|>` before teacher scoring.
- `CWOPDTrainer` (`verl/trainer/ppo/cw_opd/cw_opd_trainer.py`): subclasses `OnPolicyDistillTrainer` and overrides `_compute_ref_log_prob` so that, before the teacher pass, it computes per-token consequence weights and writes them into `batch.batch["cw_weights"]` (and also `rollout_is_weights`).

Distillation runs use `algorithm.adv_estimator=on_policy`. The "advantage" is computed inside the policy loss from teacher vs. student log-probs, not by a separate estimator.

### Policy losses

Losses are registered by name via `@register_policy_loss` in `core_algos.py` and selected with `actor_rollout_ref.actor.policy_loss.loss_mode`:
- `on_policy_distill`: upstream-fork OPD/EOPD (entropy-gated soft KD with teacher top-k), in `core_algos.py`.
- `cw_opd`, `cw_eopd`, `fuw`: in `verl/trainer/ppo/cw_opd/policy_loss.py`. These are registered as a side effect of `import verl.trainer.ppo.cw_opd` at the **bottom of `core_algos.py`**, so don't remove that import.

`verl/workers/actor/dp_actor.py` (`update_policy`) is where batch keys are routed into the loss. Its lists of which `loss_mode`s need `ref_log_prob`, `ref_entropy`, or teacher/student top-k log-probs, and its forwarding of `cw_weights` as an extra loss kwarg, are hard-coded. **A new loss mode has to be added there as well as registered.**

### CW-OPD components (`verl/trainer/ppo/cw_opd/`)

- `candidate_manager.py` maintains the candidate set C_x = {a_T} ∪ H_x per prompt (keyed by prompt hash). a_T comes from the cached greedy teacher answers. H_x is filled from student final answers after each rollout, so it grows across epochs. It also handles `\boxed{}` extraction, canonicalization, and symbolic equivalence.
- `probe_scorer.py` splits responses into paragraph steps, probes the teacher's answer distribution q_k over C_x ∪ {⊥} after each step, computes shifts g_k = TV(q_k, q_{k-1}) (or KL for an ablation), and turns them into batch-normalized token weights w̃_t (`compute_consequence_weights`). `cw_opd.ablation_mode` selects among `none | at_only | length_norm | kl_variant | permutation`.
- `teacher_probe.py` does the probing on token ids (no decode/re-encode): `segment_token_ids` (paragraph steps with exact token boundaries), `probe_batch` (all K+1 boundaries x candidates), with the teacher injected as a `score_fn`. `CWOPDTrainer._teacher_score_fn` backs it with the teacher worker group (`ref_policy_wg.compute_ref_log_prob` on probe sequences). Without a candidate set a prompt gets `g_k = 0` (weights 1) and is counted in `cw/frac_prompts_no_candidates`. `ProbeScorer.score_candidates_batch(teacher_model=None)` now raises unless `allow_mock=True`.
- **Cost caveat:** every (boundary, candidate) is an independent teacher sequence (no KV sharing / tree mask), roughly `(K+1)*|C_x|` extra sequences per rollout. Use `CW_MAX_STEPS` (`cw_opd.max_steps`) to cap K. The GPU path has not been run in this repo's CI (only CPU tests with a fake teacher).
- The a_T cache and the trainer both key prompts with `get_prompt_hash(question)`; the trainer recovers the question from the chat-templated prompt (`extract_question_from_chat_prompt`).
- Logging: scalar metrics `cw/*` go through the normal trainer logger (hook `_consume_extra_metrics` in `RayPPOTrainer.fit`); per-rollout traces (beliefs, shifts, weights) are written to `<log_dir>/probe_stepNNNNNN.jsonl` every `cw_opd.dump_every` steps.

### Hydra config keys

`trainer.trainer_class`, `actor_rollout_ref.teacher_model.*`, and `cw_opd.*`, `actor_rollout_ref.teacher_model.path` and `actor_rollout_ref.ref.topk_logits` are read with `.get()` and are **not declared** in `ppo_trainer.yaml`, so launch scripts pass them with a `+` prefix (all training scripts share `scripts/common.sh`; `DRY_RUN=1 bash scripts/run_train_*.sh` prints the resolved command). EOPD's tau/coef come from the loss defaults (0.8 / 1.0). After any YAML config change, regenerate `_generated_*.yaml` with `scripts/generate_trainer_config.sh` (a pre-commit hook verifies them).

### Standalone research scripts

- `phase0/run_phase0_pilot.py` runs the no-training pilot: signal calibration, ΔR², clustered bootstrap, martingale gap, overhead, and an automatic go/no-go. It writes `experiments/phase0_pilot/phase0_report.md`.
- `eval/evaluate_benchmarks.py` does the benchmark evaluation. `scripts/prepare_eval_benchmarks.py` and `scripts/cache_teacher_answers.py` handle data prep and the a_T cache.
