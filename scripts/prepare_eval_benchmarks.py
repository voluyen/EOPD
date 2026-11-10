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
Prepare all evaluation benchmark datasets in verl parquet format.

Benchmarks specified in Section 6:
- MATH500
- AIME24
- AIME25
- AMC23
- Minerva Math
- OlympiadBench
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datasets
import pandas as pd

from verl.trainer.ppo.cw_opd.candidate_manager import canonicalize_answer, extract_boxed_content

INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."


def process_example(question: str, solution_or_answer: str, data_source: str, idx: int) -> dict:
    prompt_content = question.strip()
    if "\\boxed{}" not in prompt_content:
        prompt_content = prompt_content + " " + INSTRUCTION_FOLLOWING

    boxed = extract_boxed_content(solution_or_answer)
    ground_truth = canonicalize_answer(boxed if boxed else solution_or_answer)

    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt_content}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {"index": idx, "data_source": data_source},
    }


def prepare_math500(output_dir: str):
    print("\n[DataPrep] Preparing MATH500...")
    try:
        ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        records = []
        for i, item in enumerate(ds):
            records.append(process_example(item["problem"], item["solution"], "math500", i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, "math500.parquet"))
        print(f"  MATH500: Saved {len(records)} examples.")
    except Exception as e:
        print(f"  Warning: Could not fetch MATH500 ({e}).")


def prepare_aime24(output_dir: str):
    print("\n[DataPrep] Preparing AIME24...")
    try:
        ds = datasets.load_dataset("math-ai/aime24", split="test")
        records = []
        for i, item in enumerate(ds):
            sol = item.get("solution", item.get("answer", ""))
            records.append(process_example(item["problem"], sol, "aime24", i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, "aime24.parquet"))
        print(f"  AIME24: Saved {len(records)} examples.")
    except Exception as e:
        print(f"  Warning: Could not fetch AIME24 ({e}).")


def prepare_aime25(output_dir: str):
    print("\n[DataPrep] Preparing AIME25...")
    try:
        ds = datasets.load_dataset("math-ai/aime25", split="test")
        records = []
        for i, item in enumerate(ds):
            sol = item.get("solution", item.get("answer", ""))
            records.append(process_example(item["problem"], sol, "aime25", i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, "aime25.parquet"))
        print(f"  AIME25: Saved {len(records)} examples.")
    except Exception as e:
        print(f"  Warning: Could not fetch AIME25 ({e}).")


def prepare_olympiadbench(output_dir: str):
    print("\n[DataPrep] Preparing OlympiadBench...")
    try:
        ds = datasets.load_dataset("open-compass/OlympiadBench", "OE_TO_maths_en_COMP", split="test")
        records = []
        for i, item in enumerate(ds):
            q = item.get("question", item.get("problem", ""))
            sol = item.get("solution", item.get("final_answer", [""])[0] if isinstance(item.get("final_answer"), list) else str(item.get("final_answer", "")))
            records.append(process_example(q, sol, "olympiadbench", i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, "olympiadbench.parquet"))
        print(f"  OlympiadBench: Saved {len(records)} examples.")
    except Exception as e:
        print(f"  Warning: Could not fetch OlympiadBench ({e}).")


def main():
    parser = argparse.ArgumentParser(description="Prepare all evaluation benchmark datasets")
    parser.add_argument("--output_dir", default="data/eval_benchmarks", help="Target directory for parquets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prepare_math500(args.output_dir)
    prepare_aime24(args.output_dir)
    prepare_aime25(args.output_dir)
    prepare_olympiadbench(args.output_dir)
    print(f"\n[DataPrep] Finished benchmark dataset preparation into {args.output_dir}\n")


if __name__ == "__main__":
    main()
