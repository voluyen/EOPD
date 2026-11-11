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
            # data_source must be one verl's reward dispatcher knows: MATH500 is used for in-training validation
            records.append(process_example(item["problem"], item["solution"], "HuggingFaceH4/MATH-500", i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, "math500.parquet"))
        print(f"  MATH500: Saved {len(records)} examples.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: could not fetch MATH500 ({e}).")
        return False


def _first_available(candidates, split_names=("test", "train")):
    """Try (repo, config) candidates in order; return the first dataset that loads."""
    last = None
    for repo, cfg in candidates:
        for split in split_names:
            try:
                return datasets.load_dataset(repo, cfg, split=split) if cfg else datasets.load_dataset(repo, split=split)
            except Exception as e:  # noqa: BLE001
                last = e
    raise RuntimeError(f"none of {[c[0] for c in candidates]} could be loaded ({last})")


def _pick(item, keys, default=""):
    for k in keys:
        if k in item and item[k] not in (None, ""):
            return item[k]
    return default


def _prepare(output_dir, name, candidates, q_keys, a_keys):
    print(f"\n[DataPrep] Preparing {name}...")
    try:
        ds = _first_available(candidates)
        records = []
        for i, item in enumerate(ds):
            ans = _pick(item, a_keys)
            if isinstance(ans, list):
                ans = ans[0] if ans else ""
            records.append(process_example(str(_pick(item, q_keys)), str(ans), name, i))
        pd.DataFrame(records).to_parquet(os.path.join(output_dir, f"{name}.parquet"))
        print(f"  {name}: saved {len(records)} examples.")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR: could not prepare {name}: {e}")
        return False


def prepare_aime24(output_dir):
    return _prepare(output_dir, "aime24", [("math-ai/aime24", None), ("HuggingFaceH4/aime_2024", None)],
                    ["problem", "question"], ["solution", "answer"])


def prepare_aime25(output_dir):
    return _prepare(output_dir, "aime25", [("math-ai/aime25", None), ("opencompass/AIME2025", "AIME2025-I")],
                    ["problem", "question"], ["answer", "solution"])


def prepare_amc23(output_dir):
    return _prepare(output_dir, "amc23", [("math-ai/amc23", None), ("AI-MO/aimo-validation-amc", None)],
                    ["question", "problem"], ["answer", "solution"])


def prepare_minerva_math(output_dir):
    return _prepare(output_dir, "minerva_math", [("math-ai/minervamath", None), ("knoveleng/Minerva-Math", None)],
                    ["question", "problem"], ["answer", "solution"])


def prepare_olympiadbench(output_dir):
    return _prepare(output_dir, "olympiadbench",
                    [("math-ai/olympiadbench", None), ("Hothan/OlympiadBench", "OE_TO_maths_en_COMP")],
                    ["question", "problem"], ["final_answer", "answer", "solution"])


def main():
    parser = argparse.ArgumentParser(description="Prepare all evaluation benchmark datasets")
    parser.add_argument("--output_dir", default="data/eval_benchmarks", help="Target directory for parquets")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "math500": prepare_math500(args.output_dir),
        "aime24": prepare_aime24(args.output_dir),
        "aime25": prepare_aime25(args.output_dir),
        "amc23": prepare_amc23(args.output_dir),
        "minerva_math": prepare_minerva_math(args.output_dir),
        "olympiadbench": prepare_olympiadbench(args.output_dir),
    }
    print(f"\n[DataPrep] Summary: {results}")
    missing = [k for k, ok in results.items() if not ok]
    if missing:
        print(f"[DataPrep] MISSING benchmarks: {missing}. Fix the dataset ids (HF hub) or place parquet files manually.")
        sys.exit(1)
    print(f"[DataPrep] All 6 benchmarks written to {args.output_dir}")


if __name__ == "__main__":
    main()
