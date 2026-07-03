#!/usr/bin/env python
"""Preflight audit for the Panda full-corpus latency-recall figure.

The audit is intentionally conservative: an existing Panda artifact is usable as
full-corpus figure evidence only when it states the required train+test pool
size. Older scaling, partial-pool, or pool-unknown files are classified as
rejected/diagnostic so later stages cannot silently promote them.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_POOL_SIZE = 2_156_234
EXPECTED_TRAIN_VIDEOS = 2_150_540
EXPECTED_TEST_QUERIES = 5_694
EXPECTED_PANDA_ROWS = 37


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def nested_get(data: dict[str, Any], keys: list[str]) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def extract_pool_size(data: dict[str, Any] | None) -> int | None:
    if not data:
        return None
    candidates = [
        ["pool_size"],
        ["search_pool_size"],
        ["num_videos"],
        ["total_videos"],
        ["metadata", "pool_size"],
        ["metadata", "search_pool_size"],
        ["metadata", "num_videos"],
        ["metrics", "pool_size"],
        ["metrics", "search_pool_size"],
    ]
    for keys in candidates:
        value = nested_get(data, keys)
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


def classify_pool(path: Path, pool_size: int | None) -> tuple[str, str]:
    if pool_size == EXPECTED_POOL_SIZE:
        return "candidate_full_corpus", "declares expected Panda train+test pool"
    if pool_size is None:
        return "rejected_pool_unknown", "does not declare search-pool size"
    return "rejected_wrong_pool", f"declares pool_size={pool_size}, expected {EXPECTED_POOL_SIZE}"


def audit_csv(path: Path, *, canonical: bool) -> dict[str, Any]:
    rows = read_csv(path)
    panda_rows = [r for r in rows if (r.get("dataset") or "").upper() == "PANDA"]
    bad_rows: list[dict[str, Any]] = []
    valid_rows = 0
    for idx, row in enumerate(panda_rows):
        n_queries = row.get("n_queries")
        row_ok = n_queries in {str(EXPECTED_TEST_QUERIES), EXPECTED_TEST_QUERIES}
        if row_ok:
            valid_rows += 1
        else:
            bad_rows.append({
                "row_index": idx,
                "method": row.get("method"),
                "op_point_value": row.get("op_point_value"),
                "reason": f"n_queries={n_queries}, expected {EXPECTED_TEST_QUERIES}",
            })
    return {
        "path": str(path),
        "exists": path.is_file(),
        "kind": "canonical_figure_csv" if canonical else "auxiliary_figure_csv",
        "rows": len(rows),
        "panda_rows": len(panda_rows),
        "valid_panda_query_rows": valid_rows,
        "invalid_panda_rows": bad_rows,
    }


def audit_scaling_csv(path: Path) -> dict[str, Any]:
    rows = read_csv(path)
    pool_sizes = []
    for row in rows:
        try:
            pool_sizes.append(int(float(row.get("pool_size", ""))))
        except ValueError:
            pass
    max_pool = max(pool_sizes) if pool_sizes else None
    status, reason = classify_pool(path, max_pool)
    # Scaling curves are not the canonical 37-row figure schema even if an
    # individual pool happened to match.
    if status == "candidate_full_corpus":
        status = "rejected_scaling_schema"
        reason = "scaling CSV is not the canonical latency-recall figure schema"
    return {
        "path": str(path),
        "exists": path.is_file(),
        "kind": "panda_scaling_csv",
        "rows": len(rows),
        "max_pool_size": max_pool,
        "status": status,
        "reason": reason,
    }


def audit_json_artifact(path: Path, kind: str) -> dict[str, Any]:
    data = read_json(path)
    pool_size = extract_pool_size(data)
    status, reason = classify_pool(path, pool_size)
    metrics = (data or {}).get("metrics") if isinstance(data, dict) else None
    total_queries = metrics.get("total_queries") if isinstance(metrics, dict) else None
    return {
        "path": str(path),
        "exists": path.is_file(),
        "kind": kind,
        "pool_size": pool_size,
        "total_queries": total_queries,
        "status": status,
        "reason": reason,
    }


def collect(repo_root: Path) -> dict[str, Any]:
    canonical_csv = repo_root / "output/evaluation_results/figures/summaries/figure_data.csv"
    aux_csv = repo_root / "output/evaluation_results/figures_panda/summaries/figure_data.csv"
    scaling_csv = repo_root / "output/evaluation_results/figures_panda_scaling/summaries/figure_data.csv"

    artifacts: list[dict[str, Any]] = [
        audit_csv(canonical_csv, canonical=True),
        audit_csv(aux_csv, canonical=False),
        audit_scaling_csv(scaling_csv),
        audit_json_artifact(
            repo_root / "output/evaluation_results/ann_baseline/panda_setting2_results.json",
            "old_ann_result_json",
        ),
    ]
    for path in sorted((repo_root / "candidates").glob("panda*_candidates_t*.json")):
        artifacts.append(audit_json_artifact(path, "candidate_json"))

    canonical = artifacts[0]
    panda_rows = int(canonical["panda_rows"])
    invalid_canonical_rows = canonical["invalid_panda_rows"]
    # A stale reuse is a PANDA row already present in the canonical figure table
    # that does not satisfy the current query/full-pool evidence contract. With
    # no canonical PANDA rows, existing stale files are rejected inputs, not reuse.
    stale_artifact_reuse = len(invalid_canonical_rows)
    rejected = [a for a in artifacts if str(a.get("status", "")).startswith("rejected")]
    candidates = [a for a in artifacts if a.get("status") == "candidate_full_corpus"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "package": "2026-06-22-panda-full-corpus-figure",
        "gate": "stale_artifact_reuse == 0",
        "expected": {
            "dataset": "PANDA",
            "setting": 2,
            "train_videos": EXPECTED_TRAIN_VIDEOS,
            "test_queries": EXPECTED_TEST_QUERIES,
            "pool_size": EXPECTED_POOL_SIZE,
            "panda_rows": EXPECTED_PANDA_ROWS,
        },
        "measured": {
            "canonical_panda_rows": panda_rows,
            "missing_required_rows": max(EXPECTED_PANDA_ROWS - panda_rows, 0),
            "stale_artifact_reuse": stale_artifact_reuse,
            "rejected_artifacts": len(rejected),
            "candidate_full_corpus_artifacts": len(candidates),
        },
        "verdict": "PASS" if stale_artifact_reuse == 0 else "FAIL",
        "artifacts": artifacts,
        "next_required_phase": "P1" if stale_artifact_reuse == 0 else "P0",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report = collect(repo_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "out": str(out),
        "verdict": report["verdict"],
        "measured": report["measured"],
    }, indent=2, sort_keys=True))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
