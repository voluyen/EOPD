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
Candidate Manager for Consequence-Weighted On-Policy Distillation (CW-OPD).

Implements Section 3.2 of proposal_consequence_weighted_opd_v7:
- Offline teacher greedy answer cache (a_T)
- Per-prompt rollout history tracker (H_x) across training epochs
- Canonicalization and equivalence clustering via math-verify / symbolic checker
- Candidate set C_x = {a_T} U H_x fixed strictly before the rollout, with |C_x| <= 3
"""

import hashlib
import json
import os
import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from math_verify import parse as mv_parse
    from math_verify import verify as mv_verify
    MATH_VERIFY_AVAILABLE = True
except ImportError:
    MATH_VERIFY_AVAILABLE = False


def check_balanced_braces(s: str) -> bool:
    """Return True iff string s has properly balanced braces '{' and '}'."""
    depth = 0
    for ch in s:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content inside the last \\boxed{...} from text.
    
    Handles nested braces correctly. Drops strings with unbalanced braces.
    """
    if not text:
        return None

    idx = text.rfind("\\boxed{")
    if idx < 0:
        # Fallback to general patterns if no boxed exists
        # e.g., "The answer is X"
        return None

    i = idx + len("\\boxed{")
    depth = 1
    content_chars = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == '{':
            depth += 1
            content_chars.append(ch)
        elif ch == '}':
            depth -= 1
            if depth > 0:
                content_chars.append(ch)
        else:
            content_chars.append(ch)
        i += 1

    if depth != 0:
        # Unbalanced braces!
        return None

    content = "".join(content_chars).strip()
    return content if content else None


def canonicalize_answer(answer_str: str) -> str:
    """Normalize and canonicalize a math answer string.
    
    Removes cosmetic whitespace, trailing dots, dollar signs, and checks brace validity.
    """
    if not answer_str:
        return ""
    
    s = answer_str.strip()
    # Remove outer math delimiters if present
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    
    # Strip unnecessary LaTeX text commands
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    
    # Standardize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    
    # Remove trailing periods if not part of a decimal
    if s.endswith(".") and not re.search(r"\d+\.$", s):
        s = s[:-1].strip()
        
    return s


def are_answers_equivalent(ans1: str, ans2: str) -> bool:
    """Check mathematical or symbolic equivalence between two answer strings."""
    c1 = canonicalize_answer(ans1)
    c2 = canonicalize_answer(ans2)
    if c1 == c2:
        return True
    
    if MATH_VERIFY_AVAILABLE:
        try:
            p1 = mv_parse(f"\\boxed{{{c1}}}")
            p2 = mv_parse(f"\\boxed{{{c2}}}")
            if p1 and p2 and mv_verify(p1, p2):
                return True
        except Exception:
            pass
            
    # Direct numerical comparison fallback
    try:
        val1 = float(c1)
        val2 = float(c2)
        if abs(val1 - val2) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass

    return False


def get_prompt_hash(prompt_text: str) -> str:
    """Deterministic id of a question (whitespace-normalised), shared by the a_T cache and the trainer."""
    norm = " ".join(prompt_text.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


_USER_TURN_RE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.DOTALL)


def extract_question_from_chat_prompt(prompt_text: str) -> str:
    """Recover the raw question from a chat-templated prompt (Qwen ChatML); returns the text itself otherwise.

    The a_T cache is keyed by the dataset's raw question, whereas the trainer only sees the templated
    prompt, so both sides must reduce to the same string before hashing.
    """
    turns = _USER_TURN_RE.findall(prompt_text)
    return turns[-1].strip() if turns else prompt_text.strip()


def question_text_from_messages(messages) -> str:
    """Raw question from a dataset ``prompt`` field (list of chat messages or a plain string)."""
    if isinstance(messages, str):
        return messages
    try:
        users = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        chosen = users[-1] if users else messages[-1]
        return str(chosen.get("content", ""))
    except Exception:
        return str(messages)


class CandidateManager:
    """Manages candidate answers C_x = {a_T} U H_x for prompts.
    
    - Caches teacher answers a_T generated before training.
    - Maintains rollout history H_x across training epochs.
    - Constructs candidate set C_x for each prompt, ensuring |C_x| <= 3.
    - Excludes the current rollout's answer from C_x.
    """

    def __init__(self, teacher_cache_path: Optional[str] = None, max_candidates: int = 3):
        self.max_candidates = max_candidates
        self.teacher_cache: Dict[str, str] = {}
        # H_x: prompt_id -> list of distinct earlier answers
        self.history: Dict[str, List[str]] = {}
        
        if teacher_cache_path and os.path.exists(teacher_cache_path):
            self.load_teacher_cache(teacher_cache_path)

    def load_teacher_cache(self, path: str):
        """Load teacher greedy answer cache from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.teacher_cache.update(data)
        print(f"[CandidateManager] Loaded {len(self.teacher_cache)} teacher answers from {path}")

    def save_teacher_cache(self, path: str):
        """Save teacher greedy answer cache to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.teacher_cache, f, indent=2, ensure_ascii=False)
        print(f"[CandidateManager] Saved {len(self.teacher_cache)} teacher answers to {path}")

    def register_teacher_answer(self, prompt_id: str, answer: str):
        """Register teacher answer a_T for a prompt."""
        clean_ans = canonicalize_answer(answer)
        if clean_ans and check_balanced_braces(clean_ans):
            self.teacher_cache[prompt_id] = clean_ans

    def get_teacher_answer(self, prompt_id: str) -> Optional[str]:
        """Retrieve teacher answer a_T for prompt."""
        ans = self.teacher_cache.get(prompt_id, None)
        return None if ans in (None, "", "None") else ans

    def get_candidate_set(self, prompt_id: str, at_only: bool = False) -> List[str]:
        """Construct candidate set C_x = {a_T} U H_x fixed before rollout.
        
        Args:
            prompt_id: Identifier of the prompt.
            at_only: If True, returns C_x = {a_T} only (Ablation T2(a)).
            
        Returns:
            List of unique, canonical candidate answers. Always has a_T first if available.
            Length is at most self.max_candidates (default 3).
        """
        a_T = self.get_teacher_answer(prompt_id)
        candidates: List[str] = []

        if a_T and a_T != "None":
            candidates.append(a_T)
        else:
            # If teacher answer not cached, use a placeholder or empty
            pass

        if at_only:
            return candidates

        # Add distinct earlier answers from H_x
        earlier_answers = self.history.get(prompt_id, [])
        for ans in earlier_answers:
            if len(candidates) >= self.max_candidates:
                break
            # Check if this answer is already in candidates (or mathematically equivalent)
            is_dup = False
            for existing in candidates:
                if are_answers_equivalent(ans, existing):
                    is_dup = True
                    break
            if not is_dup and check_balanced_braces(ans):
                candidates.append(ans)

        return candidates

    def update_history(self, prompt_id: str, student_rollout_response: str) -> Optional[str]:
        """Extract the final answer of a completed rollout and add it to H_x (used from the NEXT epoch on)."""
        boxed = extract_boxed_content(student_rollout_response)
        if not boxed:
            return None
        return self.update_history_answer(prompt_id, canonicalize_answer(boxed))

    def update_history_answer(self, prompt_id: str, clean_ans: str) -> Optional[str]:
        """Add an already extracted+canonicalised answer to H_x (deduplicated, balanced braces only)."""
        if not clean_ans or not check_balanced_braces(clean_ans):
            return None
        hist = self.history.setdefault(prompt_id, [])
        for existing in hist:
            if are_answers_equivalent(clean_ans, existing):
                return clean_ans
        hist.append(clean_ans)
        return clean_ans

    def save_history(self, path: str):
        """Persist H_x so a resumed run keeps its candidate sets."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False)
        os.replace(tmp, path)

    def load_history(self, path: str):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.history.update(json.load(f))
            print(f"[CandidateManager] Restored H_x for {len(self.history)} prompts from {path}")
