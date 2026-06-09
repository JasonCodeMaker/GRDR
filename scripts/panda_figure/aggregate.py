#!/usr/bin/env python
"""Aggregate per-(method, distractor_n) outputs into the figure CSV.

For each (method, d) it reads two artifacts produced by the pipeline:
  - stage-1 candidate JSON  -> stage1_gt_visible (recall ceiling) + avg_candidates
  - rerank JSON             -> R@1 / R@5 / R@10 / MedR / MeanR  (the y-axis)

Writes summaries/figure_data.csv, one row per (method, d). Missing rerank ->
status='missing'; a method may also drop a status.txt (e.g. 'oom') beside its
candidate dir which is surfaced verbatim. The notebook reads this CSV.

Layout convention (written by run_stage1.sh / rerank.sh):
  ${CAND_ROOT}/<method>/<method>_d<d>_candidates.json
  ${RERANK_ROOT}/<method>/d<d>/rerank.json   = {"metrics": {"R@1":..,"R@5":..,"R@10":..}}
  ${RERANK_ROOT}/<method>/d<d>/status.txt    (optional: ok|oom|skipped|...)
"""
import argparse
import csv
import glob
import json
import os

COLUMNS = [
    "method", "distractor_n", "pool_size", "n_test", "seed",
    "R@1", "R@5", "R@10", "MedR", "MeanR",
    "stage1_gt_visible", "avg_candidates", "avg_candidates_reranked",
    "candidate_file", "rerank_file", "status",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cand_root", required=True)
    p.add_argument("--rerank_root", required=True)
    p.add_argument("--manifest_dir", required=True, help="dir with panda_pool_summary.json")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--distractors", type=int, nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", required=True)
    return p.parse_args()


def load_pool_sizes(manifest_dir):
    """distractor_n -> pool_size from the manifest summary."""
    path = os.path.join(manifest_dir, "panda_pool_summary.json")
    if not os.path.exists(path):
        return {}
    summary = json.load(open(path))
    return {leg["distractor_n"]: leg["pool_size"] for leg in summary.get("legs", [])}


def stage1_stats(cand_path):
    """Compute (gt_visible_fraction, avg_candidates) from a candidate JSON."""
    if not cand_path or not os.path.exists(cand_path):
        return None, None
    data = json.load(open(cand_path))
    results = data.get("results", [])
    if not results:
        # No per-query results; fall back to recorded metrics if present.
        m = data.get("metrics", {})
        return m.get("CanHit@300") or m.get("FullSetHit@All"), m.get("avg_candidates_per_query")
    gt_in = sum(1 for r in results
                if r.get("ground_truth_video_id") in (r.get("candidates") or []))
    avg_c = sum(len(r.get("candidates") or []) for r in results) / len(results)
    return gt_in / len(results), avg_c


def rerank_metrics(rerank_dir):
    """Return (metrics_dict, rerank_file, status) for one (method, d)."""
    status_path = os.path.join(rerank_dir, "status.txt")
    status = open(status_path).read().strip() if os.path.exists(status_path) else None
    rj = os.path.join(rerank_dir, "rerank.json")
    if os.path.exists(rj):
        m = json.load(open(rj)).get("metrics", {})
        return m, rj, status or "ok"
    # Fall back to an X-Pool result.csv if rerank.json wasn't normalized.
    csvs = glob.glob(os.path.join(rerank_dir, "*.csv"))
    if csvs:
        rows = list(csv.DictReader(open(csvs[0])))
        if rows:
            last = rows[-1]
            m = {}
            for src, dst in (("R1", "R@1"), ("R5", "R@5"), ("R10", "R@10"),
                             ("R@1", "R@1"), ("R@5", "R@5"), ("R@10", "R@10"),
                             ("MedR", "MedR"), ("MeanR", "MeanR")):
                if src in last and last[src] not in (None, ""):
                    m[dst] = float(last[src])
            return m, csvs[0], status or "ok"
    return {}, None, status or "missing"


def find_candidate(cand_root, method, d):
    primary = os.path.join(cand_root, method, f"{method}_d{d}_candidates.json")
    if os.path.exists(primary):
        return primary
    hits = glob.glob(os.path.join(cand_root, method, f"*d{d}*candidates*.json"))
    return hits[0] if hits else primary


def main():
    args = parse_args()
    pool_sizes = load_pool_sizes(args.manifest_dir)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for method in args.methods:
        for d in args.distractors:
            cand = find_candidate(args.cand_root, method, d)
            gt_vis, avg_c = stage1_stats(cand)
            rerank_dir = os.path.join(args.rerank_root, method, f"d{d}")
            metrics, rfile, status = rerank_metrics(rerank_dir)
            rows.append({
                "method": method,
                "distractor_n": d,
                "pool_size": pool_sizes.get(d, args.__dict__.get("n_test", "")),
                "n_test": 5694,
                "seed": args.seed,
                "R@1": metrics.get("R@1", ""),
                "R@5": metrics.get("R@5", ""),
                "R@10": metrics.get("R@10", ""),
                "MedR": metrics.get("MedR", ""),
                "MeanR": metrics.get("MeanR", ""),
                "stage1_gt_visible": round(gt_vis, 4) if gt_vis is not None else "",
                "avg_candidates": round(avg_c, 2) if avg_c is not None else "",
                "avg_candidates_reranked": round(metrics["avg_candidates_used"], 2)
                    if "avg_candidates_used" in metrics else "",
                "candidate_file": cand if os.path.exists(cand) else "",
                "rerank_file": rfile or "",
                "status": status,
            })

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")
    for r in rows:
        print(f"  {r['method']:<12} d={r['distractor_n']:<8} pool={r['pool_size']:<10} "
              f"R@10={r['R@10']!s:<7} gt_vis={r['stage1_gt_visible']!s:<7} status={r['status']}")


if __name__ == "__main__":
    main()
