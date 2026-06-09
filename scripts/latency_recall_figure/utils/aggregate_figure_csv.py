#!/usr/bin/env python
"""Walks Pass-A + Pass-B trees and writes summaries/figure_data.{csv,json}."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


METHODS = ["grdr_ref", "tiger", "avg", "t2vindexer", "eercf", "hnsw", "ivf"]
DATASETS = ["MSRVTT", "ACTNET", "DIDEMO", "LSMDC"]
DEFAULT_OPS = {
    "grdr_ref":   [20, 50, 100, 200, 300],
    "tiger":      [20, 50, 100, 200, 300],
    "avg":        [20, 50, 100, 200, 300],
    "t2vindexer": [20, 50, 100, 200, 300],
    "eercf":      [50],
    "hnsw":       [20, 40, 100, 200, 300],
    "ivf":        [20, 40, 100, 200, 300],
}
OP_KNOB = {
    "grdr_ref":   "beam",
    "tiger":      "beam",
    "avg":        "beam",
    "t2vindexer": "budget",
    "eercf":      "rerantopk",
    "hnsw":       "budget",
    "ivf":        "budget",
}

CSV_COLUMNS = [
    "dataset", "setting", "method",
    "op_point_label", "op_point_knob", "op_point_value",
    "stage1_latency_ms", "stage2_latency_ms", "total_latency_ms",
    "CanHit@20", "CanHit@50", "CanHit@100",
    "R@1", "R@5", "R@10",
    "n_queries", "avg_candidates_per_query",
    "latency_validity", "effectiveness_validity",
    "stage1_source_path", "rerank_source_path",
]


def load_json(path: Path) -> dict[str, Any] | None:
    """Best-effort JSON load; returns None on missing/unparseable."""
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


# EERCF Stage-1 latency is the unoptimized full-pipeline retrieval cost; it is not
# re-measured in this package. Canonical per-(dataset,setting) values are hardcoded
# from a prior stage-1 latency measurement (constant; not a path dependency)
# (EERCF rerantopk=50 rows) per user directive 2026-05-27. EERCF has no separate
# X-Pool rerank stage, so total_latency == stage1_latency.
EERCF_STAGE1_LATENCY_MS = {
    # Setting-2 values are the paper-reported EERCF latencies (per user, 2026-06-03).
    ("msrvtt", 1): 2181.95, ("msrvtt", 2): 4843.0,
    ("didemo", 1): 1889.35, ("didemo", 2): 4329.0,
    ("actnet", 1): 5179.73, ("actnet", 2): 7628.0,
    ("lsmdc", 1): 2554.68,  ("lsmdc", 2): 27252.0,
}


def recompute_r1(cand_json: dict[str, Any]) -> float | None:
    """R@1 = fraction of queries whose rank-1 deduplicated candidate is the GT video.

    Used for native-retrieval baselines (EERCF) that have no X-Pool rerank artifact.
    """
    results = cand_json.get("results")
    if not results:
        return None
    hit, total = 0, 0
    for row in results:
        gt = row.get("ground_truth_video_id")
        cand_list = row.get("candidates") or []
        top1 = next(iter(dict.fromkeys(cand_list)), None)  # first unique candidate
        total += 1
        if top1 is not None and top1 == gt:
            hit += 1
    return 100.0 * hit / total if total else None


def recompute_canhit(cand_json: dict[str, Any], ks: list[int]) -> dict[int, float] | None:
    """Recomputes CanHit@K on the full result list, deduplicating candidates per row."""
    results = cand_json.get("results")
    if not results:
        return None
    counts = {k: 0 for k in ks}
    total = 0
    for row in results:
        gt = row.get("ground_truth_video_id")
        cand_list = row.get("candidates") or []
        # Deduplicate preserving order.
        seen = set()
        dedup: list[Any] = []
        for c in cand_list:
            if c in seen:
                continue
            seen.add(c)
            dedup.append(c)
        total += 1
        for k in ks:
            if gt in dedup[:k]:
                counts[k] += 1
    if total == 0:
        return None
    return {k: 100.0 * counts[k] / total for k in ks}


def candidates_a_file(cand_grdr_root: Path, cand_base_root: Path, method: str, ds: str, setting: int, op: int) -> Path:
    """Recall-Stage candidate JSON path. grdr_ref -> candidates/GRDR/<ds>/; else candidates/baselines/<method>/<ds>/."""
    ds_low = ds.lower()
    base = (cand_grdr_root / ds_low) if method == "grdr_ref" else (cand_base_root / method / ds_low)
    # Try the per-baseline naming variants the Recall-Stage cells write.
    candidates = [
        base / f"{ds_low}_t{setting}_{op}_candidates.json",
        base / f"{ds_low}_{method}_{op}_candidates_t{setting}.json",
        base / f"{ds_low}_ann_{method}_{op}_candidates_t{setting}.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]  # canonical name for missing-row reporting


def candidates_b_file(runtime_root: Path, method: str, ds: str, setting: int, op: int) -> Path:
    """Recall-Latency JSON path (stage-1 per-query latency)."""
    return runtime_root / "recall-latency" / method / ds.lower() / f"{ds.lower()}_t{setting}_{op}_latency.json"


def results_a_file(runtime_root: Path, method: str, ds: str, setting: int, op: int) -> tuple[Path, Path]:
    """Rerank-Stage X-Pool output. Returns (xpool_eval.json, result.csv) candidate paths."""
    base = runtime_root / "rerank-stage" / method / ds.lower() / f"setting{setting}" / str(op)
    return base / "xpool_eval.json", base / "result.csv"


def results_b_file(runtime_root: Path, method: str, ds: str, setting: int, op: int) -> Path:
    """Rerank-Latency per-query summary path (stage-2 latency)."""
    return runtime_root / "rerank-latency" / method / ds.lower() / f"setting{setting}" / str(op) / "perquery_summary.json"


def read_rerank_metrics(json_path: Path, csv_path: Path) -> dict[str, float | None]:
    """Returns R@1/5/10 from xpool_eval.json if present, else result.csv."""
    out: dict[str, float | None] = {"R@1": None, "R@5": None, "R@10": None}
    j = load_json(json_path)
    if j is not None:
        metrics = j.get("metrics") or {}
        for k in ("R@1", "R@5", "R@10"):
            if k in metrics:
                out[k] = float(metrics[k])
        if any(v is not None for v in out.values()):
            return out
    # Fallback: result.csv (header row + values row).
    if csv_path.is_file():
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if row:
                for k in ("R@1", "R@5", "R@10"):
                    if k in row:
                        try:
                            out[k] = float(row[k])
                        except (TypeError, ValueError):
                            pass
        except OSError:
            pass
    return out


def stage1_latency_ms(cand_b: dict[str, Any] | None) -> tuple[float | None, str]:
    """Extracts (mean_ms, validity) from Pass-B candidate JSON metadata."""
    if not cand_b:
        return None, "missing"
    meta = cand_b.get("metadata") or {}
    blk = meta.get("stage1_latency_ms") or {}
    mean = blk.get("online_total_mean")
    validity = blk.get("validity") or "missing"
    if mean is None:
        # Some baselines write per_query_timing.total_mean_ms (ANN path).
        pqt = meta.get("per_query_timing") or {}
        mean = pqt.get("total_mean_ms") or pqt.get("online_total_mean")
        validity = pqt.get("validity") or validity
    if mean is None:
        return None, "missing"
    return float(mean), str(validity)


def stage2_latency_ms(perq: dict[str, Any] | None) -> tuple[float | None, str]:
    """Extracts (mean_ms, validity) from Pass-B per-query summary."""
    if not perq:
        return None, "missing"
    blk = perq.get("rerank_latency_ms") or perq.get("summary") or {}
    mean = blk.get("online_total_mean")
    validity = blk.get("validity") or "full_subset"
    if mean is None:
        # X-Pool test_perquery.py writes summary.total_ms_mean.
        summary = perq.get("summary") or {}
        mean = summary.get("total_ms_mean")
    if mean is None:
        return None, "missing"
    return float(mean), str(validity)


def effectiveness_validity(cand_a: dict[str, Any] | None) -> str:
    """Maps Pass-A candidate JSON to {full_test_set, OOM, missing, failed}."""
    if not cand_a:
        return "missing"
    meta = cand_a.get("metadata") or {}
    if str(meta.get("status", "")).upper() == "OOM":
        return "OOM"
    if cand_a.get("results"):
        return "full_test_set"
    return "failed"


def collect(runtime_root: Path, cand_grdr_root: Path, cand_base_root: Path, methods: list[str], datasets: list[str],
            settings: list[int], ops_override: list[int] | None) -> list[dict[str, Any]]:
    """Walks the grid, returning one row dict per (method, ds, setting, op) cell."""
    rows: list[dict[str, Any]] = []
    for method in methods:
        ops = ops_override if ops_override else DEFAULT_OPS.get(method, [])
        knob = OP_KNOB[method]
        for ds in datasets:
            for setting in settings:
                for op in ops:
                    cand_a = load_json(candidates_a_file(cand_grdr_root, cand_base_root, method, ds, setting, op))
                    cand_b = load_json(candidates_b_file(runtime_root, method, ds, setting, op))
                    perq = load_json(results_b_file(runtime_root, method, ds, setting, op))
                    rerank_json, rerank_csv = results_a_file(runtime_root, method, ds, setting, op)
                    rerank = read_rerank_metrics(rerank_json, rerank_csv)

                    canhit = recompute_canhit(cand_a, [20, 50, 100]) if cand_a else None
                    s1_ms, latency_validity = stage1_latency_ms(cand_b)
                    s2_ms, _ = stage2_latency_ms(perq)
                    total_ms = None
                    if s1_ms is not None and s2_ms is not None:
                        total_ms = s1_ms + s2_ms

                    if method == "eercf":
                        # Stage-1 = hardcoded full-pipeline retrieval; no separate
                        # X-Pool rerank, so total == stage1. The measured cand_b value
                        # is the optimized top-K rerank (not the cited Stage-1 cost).
                        hard_s1 = EERCF_STAGE1_LATENCY_MS.get((ds.lower(), int(setting)))
                        if hard_s1 is not None:
                            s1_ms = hard_s1
                            s2_ms = None
                            total_ms = hard_s1
                            latency_validity = "hardcoded_stage1_2026-05-06"
                        # EERCF has no X-Pool rerank artifact: R@1 from its own candidates.
                        if rerank.get("R@1") is None and cand_a:
                            r1 = recompute_r1(cand_a)
                            if r1 is not None:
                                rerank = {**rerank, "R@1": r1}

                    cand_a_metrics = (cand_a or {}).get("metrics") or {}
                    n_queries = cand_a_metrics.get("total_queries")
                    avg_cands = cand_a_metrics.get("avg_candidates_per_query")

                    eff_validity = effectiveness_validity(cand_a)

                    s1_src = candidates_b_file(runtime_root, method, ds, setting, op)
                    rerank_src = rerank_json if rerank_json.is_file() else rerank_csv

                    row = {
                        "dataset": ds,
                        "setting": setting,
                        "method": method,
                        "op_point_label": f"{knob}={op}",
                        "op_point_knob": knob,
                        "op_point_value": op,
                        "stage1_latency_ms": s1_ms,
                        "stage2_latency_ms": s2_ms,
                        "total_latency_ms": total_ms,
                        "CanHit@20":  canhit[20] if canhit else None,
                        "CanHit@50":  canhit[50] if canhit else None,
                        "CanHit@100": canhit[100] if canhit else None,
                        "R@1":  rerank.get("R@1"),
                        "R@5":  rerank.get("R@5"),
                        "R@10": rerank.get("R@10"),
                        "n_queries": n_queries,
                        "avg_candidates_per_query": avg_cands,
                        "latency_validity": latency_validity,
                        "effectiveness_validity": eff_validity,
                        "stage1_source_path": os.path.relpath(s1_src, runtime_root),
                        "rerank_source_path": os.path.relpath(rerank_src, runtime_root),
                    }
                    rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes the canonical 21-column CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: ("" if row.get(col) is None else row[col]) for col in CSV_COLUMNS})


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    """Writes the same rows as a JSON list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, indent=2)


def main() -> int:
    """Aggregates Pass-A + Pass-B into figure_data.csv and runs lint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime_root", required=True)
    ap.add_argument("--cand_grdr_root", required=True)
    ap.add_argument("--cand_base_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--methods", default=" ".join(METHODS))
    ap.add_argument("--datasets", default=" ".join(DATASETS))
    ap.add_argument("--settings", default="2")
    ap.add_argument("--operating_points", default="")
    args = ap.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    cand_grdr_root = Path(args.cand_grdr_root).resolve()
    cand_base_root = Path(args.cand_base_root).resolve()
    methods = args.methods.split()
    datasets = args.datasets.split()
    settings = [int(s) for s in args.settings.split()]
    ops_override = [int(o) for o in args.operating_points.split()] if args.operating_points.strip() else None

    rows = collect(runtime_root, cand_grdr_root, cand_base_root, methods, datasets, settings, ops_override)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)
    write_csv(rows, out_csv)
    write_json(rows, out_json)
    print(f"wrote {len(rows)} rows to {out_csv}")

    # Auto-invoke lint at end.
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import lint_figure_data
        lint_figure_data.main_with_args(csv_path=out_csv,
                                        runtime_root=runtime_root)
    except Exception as e:
        print(f"lint invocation failed: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
