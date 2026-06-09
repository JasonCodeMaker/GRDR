#!/usr/bin/env python
"""Compute Candidate-Pool Recall vs beam size per method/dataset (Fig 3).

Candidate-Pool Recall = % of queries whose ground-truth video appears anywhere
in the Stage-1 candidate pool handed to the reranker (i.e. GT in the dedup pool;
equivalently FullSetHit). It upper-bounds the reranker's R@K (R@K <= pool recall).

The x-axis is the decode **beam size** (the `num_candidates` op used at generation
time, recovered from the candidate-JSON filename `{ds}_t2_{beam}_latency.json`).
For one-to-one GR (each video has a single sID -> single route) the dedup pool
size equals the beam; for GRDR the multi-route expansion makes the pool grow past
the beam, so more GT videos become reachable at the same decode work.

Writes a long CSV (method, dataset, beam, pool_recall, mean_pool, n_queries).
The candidate lists are the on-disk ~200q latency subsets.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

REPO = "/home/uqzzha35/Project/SemanticID/GRDR"
RL_ROOT = f"{REPO}/output/evaluation_results/figures/recall-latency"

# (json subdir, plot label)
METHODS = [
    ("grdr_ref", "GRDR"),
    ("t2vindexer", "T2VIndexer"),
    ("tiger", "TIGER"),
    ("avg", "AVG"),
]
DATASETS = [("msrvtt", "MSR-VTT"), ("actnet", "ActivityNet"),
            ("didemo", "DiDeMo"), ("lsmdc", "LSMDC")]


def beam_files(method, ds):
    """Return [(beam, path), ...] for every beam-swept candidate JSON on disk."""
    out = []
    for p in glob.glob(f"{RL_ROOT}/{method}/{ds}/{ds}_t2_*_latency.json"):
        m = re.search(rf"{ds}_t2_(\d+)_latency\.json$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REPO}/output/evaluation_results/figures/summaries/coverage_data.csv")
    args = ap.parse_args()

    rows = []
    for ds, ds_name in DATASETS:
        for method, label in METHODS:
            files = beam_files(method, ds)
            if not files:
                print(f"  skip (no files): {method}/{ds}")
                continue
            for beam, p in files:
                res = json.load(open(p))["results"]
                mean_pool = np.mean([len(r["candidates"]) for r in res])
                pool_recall = np.mean(
                    [r["ground_truth_video_id"] in r["candidates"] for r in res]
                ) * 100
                rows.append((label, ds_name, beam, pool_recall, mean_pool, len(res)))
                print(f"  {label:>11} / {ds_name:<11} beam={beam:>4}: "
                      f"pool_recall={pool_recall:5.1f}  mean_pool={mean_pool:6.0f}  nq={len(res)}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,dataset,beam,pool_recall,mean_pool,n_queries\n")
        for label, ds_name, beam, pr, mp, nq in rows:
            f.write(f"{label},{ds_name},{beam},{pr:.2f},{mp:.1f},{nq}\n")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
