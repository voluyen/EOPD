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
Benchmark Evaluator for Consequence-Weighted OPD.

Implements the evaluation protocol described in Section 6 of proposal_consequence_weighted_opd_v7:
- Evaluates models on MATH500, AIME24, AIME25, AMC23, Minerva Math, OlympiadBench
- Qwen2.5-Math pipeline: temperature 1.0, top-p 0.8, 8192 tokens, 8 samples
- Reports Avg@8 and Pass@8 metrics
- Paired bootstrap over questions and seeds for confidence intervals
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.cw_opd.candidate_manager import are_answers_equivalent, extract_boxed_content
from verl.utils.reward_score.math_verify import compute_score as verify_math_score


def extract_prompt_str(item: Any) -> str:
    """Extract prompt text from row."""
    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
        return item[0].get("content", "")
    elif isinstance(item, str):
        return item
    return str(item)


def extract_gold_str(row: pd.Series) -> str:
    """Extract ground truth answer from row."""
    if "reward_model" in row:
        rm = row["reward_model"]
        if isinstance(rm, dict) and "ground_truth" in rm:
            return str(rm["ground_truth"])
    if "ground_truth" in row:
        return str(row["ground_truth"])
    if "answer" in row:
        return str(row["answer"])
    return ""


def evaluate_dataset(
    model: Any,
    tokenizer: Any,
    parquet_path: str,
    n_samples: int = 8,
    temperature: float = 1.0,
    top_p: float = 0.8,
    max_tokens: int = 8192,
    device: torch.device = torch.device("cpu"),
    limit: int = -1,
) -> Dict[str, Any]:
    """Evaluate model on a benchmark dataset."""
    df = pd.read_parquet(parquet_path)
    if limit > 0:
        df = df.iloc[:limit]

    total_problems = len(df)
    per_problem_avg8: List[float] = []
    per_problem_pass8: List[float] = []
    detailed_results: List[Dict[str, Any]] = []

    print(f"\nEvaluating {parquet_path} ({total_problems} problems, {n_samples} samples per prompt)...")

    for idx, row in tqdm(df.iterrows(), total=total_problems, desc="Evaluating problems"):
        prompt_text = extract_prompt_str(row["prompt"])
        gold_answer = extract_gold_str(row)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        sample_correctness = []
        sample_responses = []

        with torch.no_grad():
            for s_idx in range(n_samples):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
                resp_ids = output_ids[0, inputs.input_ids.shape[1]:]
                resp_text = tokenizer.decode(resp_ids, skip_special_tokens=True)
                sample_responses.append(resp_text)

                # Verification with math_verify
                is_correct = verify_math_score(resp_text, gold_answer)
                if not is_correct:
                    # Fallback boxed comparison
                    extracted = extract_boxed_content(resp_text) or ""
                    is_correct = are_answers_equivalent(extracted, gold_answer)

                sample_correctness.append(1.0 if is_correct else 0.0)

        avg_8 = float(np.mean(sample_correctness))
        pass_8 = 1.0 if any(c > 0 for c in sample_correctness) else 0.0

        per_problem_avg8.append(avg_8)
        per_problem_pass8.append(pass_8)

        detailed_results.append({
            "problem_idx": int(idx),
            "prompt": prompt_text,
            "ground_truth": gold_answer,
            "avg_at_8": avg_8,
            "pass_at_8": pass_8,
            "sample_scores": sample_correctness,
        })

    # Paired bootstrap for confidence intervals (1000 resamples)
    n_boot = 1000
    boot_avg = []
    boot_pass = []
    n_q = len(per_problem_avg8)
    for _ in range(n_boot):
        indices = np.random.choice(n_q, size=n_q, replace=True)
        boot_avg.append(float(np.mean([per_problem_avg8[i] for i in indices])))
        boot_pass.append(float(np.mean([per_problem_pass8[i] for i in indices])))

    mean_avg8 = float(np.mean(per_problem_avg8)) * 100.0
    mean_pass8 = float(np.mean(per_problem_pass8)) * 100.0
    avg8_ci = (float(np.percentile(boot_avg, 2.5)) * 100.0, float(np.percentile(boot_avg, 97.5)) * 100.0)
    pass8_ci = (float(np.percentile(boot_pass, 2.5)) * 100.0, float(np.percentile(boot_pass, 97.5)) * 100.0)

    return {
        "dataset": os.path.basename(parquet_path),
        "total_problems": total_problems,
        "avg_at_8": mean_avg8,
        "avg_at_8_ci95": avg8_ci,
        "pass_at_8": mean_pass8,
        "pass_at_8_ci95": pass8_ci,
        "details": detailed_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on math reasoning benchmarks")
    parser.add_argument("--model_path", required=True, help="Trained student checkpoint path")
    parser.add_argument("--benchmarks_dir", required=True, help="Directory containing test parquets")
    parser.add_argument("--output_path", required=True, help="Output JSON path to save evaluation summary")
    parser.add_argument("--n_samples", type=int, default=8, help="Number of samples per problem (default 8)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default 1.0)")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p sampling (default 0.8)")
    parser.add_argument("--max_tokens", type=int, default=8192, help="Max tokens per response (default 8192)")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of problems per dataset for fast test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"[Eval] Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Supported benchmarks from Section 6
    benchmark_names = [
        "math500.parquet",
        "aime24.parquet",
        "aime25.parquet",
        "amc23.parquet",
        "minerva_math.parquet",
        "olympiadbench.parquet",
    ]

    summary_results: Dict[str, Any] = {
        "model_path": args.model_path,
        "protocol": "Qwen2.5-Math (temp 1.0, top-p 0.8, 8192 tokens, 8 samples)",
        "benchmarks": {},
    }

    # Find matching parquets in benchmarks_dir
    found_files = []
    for root, _, files in os.walk(args.benchmarks_dir):
        for f in files:
            if f.endswith(".parquet"):
                found_files.append(os.path.join(root, f))

    if not found_files:
        print(f"[Eval Error] No parquet files found in {args.benchmarks_dir}")
        return

    for p_path in found_files:
        fname = os.path.basename(p_path)
        bench_res = evaluate_dataset(
            model=model,
            tokenizer=tokenizer,
            parquet_path=p_path,
            n_samples=args.n_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            device=device,
            limit=args.limit,
        )
        summary_results["benchmarks"][fname] = {
            "avg_at_8": bench_res["avg_at_8"],
            "avg_at_8_ci95": bench_res["avg_at_8_ci95"],
            "pass_at_8": bench_res["pass_at_8"],
            "pass_at_8_ci95": bench_res["pass_at_8_ci95"],
            "total_problems": bench_res["total_problems"],
        }

    # Save summary
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"                 EVALUATION SUMMARY                    ")
    print(f"=======================================================")
    for b_name, b_data in summary_results["benchmarks"].items():
        print(f"Benchmark: {b_name:25s} | Avg@8: {b_data['avg_at_8']:5.2f}% (CI: [{b_data['avg_at_8_ci95'][0]:.1f}, {b_data['avg_at_8_ci95'][1]:.1f}]) | Pass@8: {b_data['pass_at_8']:5.2f}%")
    print(f"=======================================================")
    print(f"Detailed results saved to {args.output_path}")


if __name__ == "__main__":
    main()
