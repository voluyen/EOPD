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
Benchmark evaluation (Section 6 of proposal v7): MATH500, AIME24, AIME25, AMC23, Minerva Math, OlympiadBench.

Protocol: temperature 1.0, top-p 0.8, 8192 new tokens, 8 samples per problem -> Avg@8 and Pass@8.
Prompts go through the model's chat template (as in training). vLLM is used when available
(HF generate is a very slow fallback). Besides the summary, per-problem / per-sample scores are
written to ``<output>.per_problem.json`` so that methods can be compared with a paired bootstrap
(``eval/compare_results.py``).
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from verl.trainer.ppo.cw_opd.candidate_manager import are_answers_equivalent, extract_boxed_content

BENCHMARKS = ["math500", "aime24", "aime25", "amc23", "minerva_math", "olympiadbench"]


def row_messages(prompt) -> List[Dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    return [dict(m) for m in list(prompt)]


def row_gold(row: pd.Series) -> str:
    rm = row.get("reward_model", None)
    if isinstance(rm, dict) and "ground_truth" in rm:
        return str(rm["ground_truth"])
    for k in ("ground_truth", "answer"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def is_correct(response: str, gold: str) -> bool:
    try:
        from verl.utils.reward_score.math_verify import compute_score as mv_score

        if mv_score(response, gold):
            return True
    except Exception:  # math_verify missing or failing on this input
        pass
    extracted = extract_boxed_content(response) or ""
    return bool(extracted) and are_answers_equivalent(extracted, gold)


def chat_prompts(tokenizer, messages_list) -> List[str]:
    out = []
    for msgs in messages_list:
        try:
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=False))
        except TypeError:
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


class VLLMGenerator:
    def __init__(self, model_path, tp, gpu_util, max_tokens, seed):
        from vllm import LLM

        self.llm = LLM(model=model_path, tensor_parallel_size=tp, gpu_memory_utilization=gpu_util,
                       trust_remote_code=True, dtype="bfloat16", seed=seed,
                       max_model_len=max_tokens + 2048)

    def __call__(self, prompts, n, temperature, top_p, max_tokens, seed):
        from vllm import SamplingParams

        sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        return [[c.text for c in o.outputs] for o in self.llm.generate(prompts, sp)]


class HFGenerator:
    def __init__(self, model_path, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()

    def __call__(self, prompts, n, temperature, top_p, max_tokens, seed):
        import torch

        torch.manual_seed(seed)
        results = []
        for p in prompts:
            inp = self.tok(p, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inp, max_new_tokens=max_tokens, do_sample=True, temperature=temperature,
                                          top_p=top_p, num_return_sequences=n, pad_token_id=self.tok.pad_token_id)
            results.append(self.tok.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True))
        return results


def bootstrap_ci(per_problem: np.ndarray, n_boot: int = 1000, seed: int = 0):
    rng = np.random.RandomState(seed)
    n = len(per_problem)
    means = [per_problem[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)) * 100, float(np.percentile(means, 97.5)) * 100


def main():
    ap = argparse.ArgumentParser(description="Evaluate a model on the six math benchmarks")
    ap.add_argument("--model_path", required=True, help="HF-format model dir (merged / hf_model checkpoint)")
    ap.add_argument("--benchmarks_dir", required=True)
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.8)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    print(f"[Eval] model={args.model_path} backend={backend}")
    gen = (VLLMGenerator(args.model_path, args.tp, args.gpu_util, args.max_tokens, args.seed)
           if backend == "vllm" else HFGenerator(args.model_path))

    summary: Dict[str, Any] = {
        "model_path": args.model_path, "seed": args.seed, "backend": backend,
        "protocol": f"temp {args.temperature}, top-p {args.top_p}, {args.max_tokens} tokens, {args.n_samples} samples",
        "benchmarks": {},
    }
    per_problem_out: Dict[str, Any] = {}
    missing = [b for b in BENCHMARKS if not os.path.exists(os.path.join(args.benchmarks_dir, f"{b}.parquet"))]
    if missing:
        print(f"[Eval WARNING] missing benchmarks (skipped): {missing}")
        summary["missing_benchmarks"] = missing

    for name in BENCHMARKS:
        path = os.path.join(args.benchmarks_dir, f"{name}.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if args.limit > 0:
            df = df.iloc[: args.limit]
        msgs = [row_messages(p) for p in df["prompt"]]
        golds = [row_gold(r) for _, r in df.iterrows()]
        print(f"[Eval] {name}: {len(df)} problems x {args.n_samples} samples")
        outs = gen(chat_prompts(tokenizer, msgs), args.n_samples, args.temperature, args.top_p, args.max_tokens,
                   args.seed)
        scores = np.array([[float(is_correct(r, g)) for r in rs] for rs, g in zip(outs, golds)])  # (problems, n)
        avg8, pass8 = scores.mean(axis=1), (scores.max(axis=1) > 0).astype(float)
        summary["benchmarks"][name] = {
            "avg_at_8": float(avg8.mean() * 100), "avg_at_8_ci95": bootstrap_ci(avg8),
            "pass_at_8": float(pass8.mean() * 100), "pass_at_8_ci95": bootstrap_ci(pass8),
            "total_problems": int(len(df)),
            "mean_response_chars": float(np.mean([len(r) for rs in outs for r in rs])),
            "frac_with_boxed": float(np.mean([extract_boxed_content(r) is not None for rs in outs for r in rs])),
        }
        per_problem_out[name] = {"avg_at_8": avg8.tolist(), "pass_at_8": pass8.tolist(), "sample_scores": scores.tolist()}
        b = summary["benchmarks"][name]
        print(f"[Eval] {name:14s} Avg@8 {b['avg_at_8']:5.2f} {tuple(round(x,1) for x in b['avg_at_8_ci95'])}  "
              f"Pass@8 {b['pass_at_8']:5.2f}  boxed {b['frac_with_boxed']:.2f}")

    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(args.output_path + ".per_problem.json", "w") as f:
        json.dump(per_problem_out, f)
    print(f"[Eval] saved {args.output_path} (+ .per_problem.json)")


if __name__ == "__main__":
    main()
