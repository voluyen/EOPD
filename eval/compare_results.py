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
Paired bootstrap between methods (Section 6: "paired bootstrap over questions and seeds").

    python eval/compare_results.py --baseline checkpoints/t1_eopd_seed{0,1,2}/eval_results.json \
        --method checkpoints/t1_ours_cw_eopd_seed{0,1,2}/eval_results.json

Inputs are the eval_results.json files (the .per_problem.json next to each is read). Seeds are paired by
position. Each bootstrap draw resamples questions (shared by both methods) and seeds, and reports the
difference of Avg@8 (method - baseline) with a 95% interval, per benchmark and averaged over benchmarks.
"""

import argparse
import json

import numpy as np


def load(paths):
    return [json.load(open(p + ".per_problem.json")) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--method", nargs="+", required=True)
    ap.add_argument("--metric", default="avg_at_8", choices=["avg_at_8", "pass_at_8"])
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    assert len(args.baseline) == len(args.method), "pair runs by seed: same number of files"
    base, meth = load(args.baseline), load(args.method)
    rng = np.random.RandomState(args.seed)
    S = len(base)

    benches = sorted(set(base[0]) & set(meth[0]))
    diffs_all = []
    print(f"{'benchmark':14s} {'baseline':>9s} {'method':>9s} {'delta':>8s}   95% paired-bootstrap CI")
    for b in benches:
        B = np.array([r[b][args.metric] for r in base])  # (seeds, questions)
        M = np.array([r[b][args.metric] for r in meth])
        n = B.shape[1]
        draws = []
        for _ in range(args.n_boot):
            q = rng.randint(0, n, n)
            s = rng.randint(0, S, S)
            draws.append((M[s][:, q] - B[s][:, q]).mean())
        diffs_all.append(draws)
        lo, hi = np.percentile(draws, [2.5, 97.5]) * 100
        print(f"{b:14s} {B.mean()*100:9.2f} {M.mean()*100:9.2f} {(M.mean()-B.mean())*100:+8.2f}   [{lo:+.2f}, {hi:+.2f}]"
              + ("  *" if lo > 0 or hi < 0 else ""))
    avg = np.mean(np.array(diffs_all), axis=0)
    lo, hi = np.percentile(avg, [2.5, 97.5]) * 100
    print(f"{'MEAN':14s} {'':9s} {'':9s} {avg.mean()*100:+8.2f}   [{lo:+.2f}, {hi:+.2f}]" + ("  *" if lo > 0 or hi < 0 else ""))


if __name__ == "__main__":
    main()
