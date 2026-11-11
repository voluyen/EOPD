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
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tqdm import tqdm

from verl.trainer.ppo.cw_opd.candidate_manager import (
    canonicalize_answer,
    check_balanced_braces,
    extract_boxed_content,
    get_prompt_hash,
    question_text_from_messages,
)


def build_chat_prompts(tokenizer, messages_list) -> List[str]:
    """Teacher prompts: the dataset's chat messages through the chat template, Qwen3 in non-thinking mode."""
    out = []
    for msgs in messages_list:
        if isinstance(msgs, str):
            msgs = [{"role": "user", "content": msgs}]
        msgs = [dict(m) for m in list(msgs)]
        try:
            out.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        except TypeError:
            out.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    return out


def generate_vllm(model_path, chat_prompts, max_tokens, tp, gpu_util):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, tensor_parallel_size=tp, gpu_memory_utilization=gpu_util,
              trust_remote_code=True, dtype="bfloat16", max_model_len=max_tokens + 2048)
    outs = llm.generate(chat_prompts, SamplingParams(temperature=0.0, max_tokens=max_tokens))  # greedy
    return [o.outputs[0].text for o in outs]


def generate_hf(model_path, tokenizer, chat_prompts, max_tokens, batch_size):
    import torch
    from transformers import AutoModelForCausalLM

    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True).eval()
    results = []
    for b in tqdm(range(0, len(chat_prompts), batch_size), desc="HF greedy"):
        batch = chat_prompts[b: b + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                 temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.pad_token_id)
        results += tokenizer.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return results


def main():
    parser = argparse.ArgumentParser(description="Cache greedy teacher answers a_T before training")
    parser.add_argument("--data_path", required=True, help="Path to preprocessed train.parquet")
    parser.add_argument("--model_path", default="Qwen/Qwen3-8B", help="Teacher model name or path")
    parser.add_argument("--output_path", required=True, help="Output JSON path to save cache")
    parser.add_argument("--backend", default="auto", choices=["auto", "vllm", "hf"])
    parser.add_argument("--batch_size", type=int, default=16, help="HF backend batch size")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens for teacher")
    parser.add_argument("--tp", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu_util", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of prompts (for testing)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    df = pd.read_parquet(args.data_path)
    if args.limit > 0:
        df = df.iloc[: args.limit]
    messages = list(df["prompt"])
    keys = [get_prompt_hash(question_text_from_messages(m)) for m in messages]
    print(f"[Cache Teacher] {len(df)} prompts; {len(set(keys))} unique questions")

    cache: Dict[str, str] = {}
    if os.path.exists(args.output_path):
        with open(args.output_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        print(f"[Cache Teacher] Resumed with {len(cache)} cached answers")

    pending = [i for i, k in enumerate(keys) if k not in cache]
    pending = list({keys[i]: i for i in pending}.values())  # one generation per unique question
    print(f"[Cache Teacher] To generate: {len(pending)}")
    if not pending:
        print("[Cache Teacher] Nothing to do.")
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chat_prompts = build_chat_prompts(tokenizer, [messages[i] for i in pending])

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401
            backend = "vllm"
        except ImportError:
            backend = "hf"
    print(f"[Cache Teacher] Generating with backend={backend} (greedy, non-thinking)")
    if backend == "vllm":
        texts = generate_vllm(args.model_path, chat_prompts, args.max_tokens, args.tp, args.gpu_util)
    else:
        texts = generate_hf(args.model_path, tokenizer, chat_prompts, args.max_tokens, args.batch_size)

    n_none = 0
    for i, text in zip(pending, texts):
        boxed = extract_boxed_content(text)
        canonical = canonicalize_answer(boxed) if boxed else ""
        if canonical and check_balanced_braces(canonical):
            cache[keys[i]] = canonical
        else:
            cache[keys[i]] = "None"  # kept so the prompt is not regenerated; treated as "no a_T" in training
            n_none += 1

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"[Cache Teacher] Saved {len(cache)} answers to {args.output_path} "
          f"({n_none} without a boxed answer -> {n_none / max(1, len(pending)):.1%})")


if __name__ == "__main__":
    main()
