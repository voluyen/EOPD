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
Offline Teacher Answer Caching for Consequence-Weighted OPD.

Generates greedy answers a_T from the teacher model for all training prompts,
canonicalizes the answers, and caches them to a JSON file before training begins.
Implements Section 3.2 of proposal_consequence_weighted_opd_v7.
"""

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.trainer.ppo.cw_opd.candidate_manager import (
    canonicalize_answer,
    extract_boxed_content,
    get_prompt_hash,
)


def extract_prompt_text(item) -> str:
    """Extract prompt text from parquet row."""
    if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
        return item[0].get("content", "")
    elif isinstance(item, str):
        return item
    return str(item)


def main():
    parser = argparse.ArgumentParser(description="Cache greedy teacher answers a_T before training")
    parser.add_argument("--data_path", required=True, help="Path to preprocessed train.parquet")
    parser.add_argument("--model_path", default="Qwen/Qwen3-8B", help="Teacher model name or path")
    parser.add_argument("--output_path", required=True, help="Output JSON path to save cache")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for teacher generation")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens for teacher")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of prompts (for testing)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"[Cache Teacher] Reading dataset from {args.data_path}...")
    df = pd.read_parquet(args.data_path)
    if args.limit > 0:
        df = df.iloc[:args.limit]

    print(f"[Cache Teacher] Total prompts to process: {len(df)}")
    prompts = [extract_prompt_text(p) for p in df["prompt"]]

    # Resume support if file already exists
    cache: Dict[str, str] = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[Cache Teacher] Resumed with {len(cache)} existing cached answers.")

    # Filter out already cached prompts
    pending_indices = [i for i, p in enumerate(prompts) if get_prompt_hash(p) not in cache]
    print(f"[Cache Teacher] Remaining prompts to generate: {len(pending_indices)}")

    if not pending_indices:
        print("[Cache Teacher] All answers already cached! Exiting.")
        return

    print(f"[Cache Teacher] Loading tokenizer and model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    # Process in batches
    for b_idx in tqdm(range(0, len(pending_indices), args.batch_size), desc="Caching teacher answers"):
        batch_ids = pending_indices[b_idx : b_idx + args.batch_size]
        batch_prompts = [prompts[i] for i in batch_ids]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=False,  # Greedy answering as required by Section 3.2
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        resp_ids = output_ids[:, inputs.input_ids.shape[1]:]
        decoded_responses = tokenizer.batch_decode(resp_ids, skip_special_tokens=True)

        for p_text, resp in zip(batch_prompts, decoded_responses):
            p_hash = get_prompt_hash(p_text)
            boxed = extract_boxed_content(resp)
            canonical = canonicalize_answer(boxed or "")
            cache[p_hash] = canonical if canonical else "None"

        # Save checkpoint periodically
        if (b_idx // args.batch_size) % 10 == 0:
            with open(args.output_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2, ensure_ascii=False)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"[Cache Teacher] Successfully cached {len(cache)} answers to {args.output_path}")


if __name__ == "__main__":
    main()
