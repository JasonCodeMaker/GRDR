#!/usr/bin/env python
"""Lints summaries/figure_data.csv against the docs/validation.html contract."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = [
    "dataset", "setting", "method",
    "op_point_label", "op_point_knob", "op_point_value",
    "stage1_latency_ms", "stage2_latency_ms", "total_latency_ms",
    "CanHit@20", "CanHit@50", "CanHit@100",
    "R@1", "R@5", "R@10",
    "n_queries", "avg_candidates_per_query",
    "latency_validity", "effectiveness_validity",
    "stage1_source_path", "rerank_source_path",
]

def _to_float(v: Any) -> float | None:
    """Tolerant float coercion: '' / None -> None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def schema_check(rows: list[dict[str, str]]) -> list[str]:
    """Verifies the CSV has the 21 required columns."""
    if not rows:
        return ["empty CSV (no data rows)"]
    missing = [c for c in REQUIRED_COLUMNS if c not in rows[0]]
    return [f"missing column: {c}" for c in missing]


def percent_bound_check(rows: list[dict[str, str]]) -> list[str]:
    """0 <= CanHit@K <= 100, 0 <= R@K <= 100 for present values."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        for col in ("CanHit@20", "CanHit@50", "CanHit@100", "R@1", "R@5", "R@10"):
            v = _to_float(row.get(col))
            if v is None:
                continue
            if not (0.0 <= v <= 100.0):
                errors.append(f"row {i} {row.get('method')}/{row.get('dataset')}/op={row.get('op_point_value')}: {col}={v} out of [0,100]")
    return errors


def latency_sign_check(rows: list[dict[str, str]]) -> list[str]:
    """stage1 > 0 and total >= stage1 for every full_subset row."""
    errors: list[str] = []
    for i, row in enumerate(rows):
        if row.get("latency_validity") != "full_subset":
            continue
        s1 = _to_float(row.get("stage1_latency_ms"))
        tot = _to_float(row.get("total_latency_ms"))
        if s1 is None or s1 <= 0:
            errors.append(f"row {i} {row.get('method')}/{row.get('dataset')}/op={row.get('op_point_value')}: stage1_latency_ms not > 0 ({s1})")
        if tot is not None and s1 is not None and tot < s1:
            errors.append(f"row {i} {row.get('method')}/{row.get('dataset')}/op={row.get('op_point_value')}: total ({tot}) < stage1 ({s1})")
    return errors


def witness_resolve_check(rows: list[dict[str, str]], runtime_root: Path) -> list[str]:
    """Warns when stage1_source_path / rerank_source_path do not resolve."""
    warnings: list[str] = []
    for i, row in enumerate(rows):
        for col in ("stage1_source_path", "rerank_source_path"):
            p = row.get(col)
            if not p:
                continue
            if not (runtime_root / p).is_file():
                warnings.append(f"row {i} {row.get('method')}/{row.get('dataset')}: {col} not on disk: {p}")
    return warnings


def monotonicity_check(rows: list[dict[str, str]]) -> list[str]:
    """Within each (dataset, setting, method), CanHit@100 and stage1 grow in op_point_value."""
    warnings: list[str] = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("dataset", ""), row.get("setting", ""), row.get("method", ""))
        groups.setdefault(key, []).append(row)
    for key, grp in groups.items():
        try:
            grp_sorted = sorted(grp, key=lambda r: int(r.get("op_point_value") or 0))
        except (TypeError, ValueError):
            continue
        last_can, last_lat = None, None
        for row in grp_sorted:
            can = _to_float(row.get("CanHit@100"))
            lat = _to_float(row.get("stage1_latency_ms"))
            if last_can is not None and can is not None and can + 1e-6 < last_can:
                warnings.append(f"{key} op={row.get('op_point_value')}: CanHit@100 non-monotonic ({last_can} -> {can})")
            if last_lat is not None and lat is not None and lat + 1e-6 < last_lat:
                warnings.append(f"{key} op={row.get('op_point_value')}: stage1_latency_ms non-monotonic ({last_lat} -> {lat})")
            if can is not None:
                last_can = can
            if lat is not None:
                last_lat = lat
    return warnings


def main_with_args(csv_path: Path, runtime_root: Path) -> int:
    """Programmatic entry used by aggregate_figure_csv."""
    csv_path = Path(csv_path)
    runtime_root = Path(runtime_root)
    if not csv_path.is_file():
        print(f"LINT ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    warnings: list[str] = []
    errors += schema_check(rows)
    if errors:
        for e in errors:
            print(f"LINT ERROR: {e}", file=sys.stderr)
        return 1

    errors += percent_bound_check(rows)
    errors += latency_sign_check(rows)
    warnings += witness_resolve_check(rows, runtime_root)
    warnings += monotonicity_check(rows)

    for w in warnings:
        print(f"LINT WARN: {w}", file=sys.stderr)
    for e in errors:
        print(f"LINT ERROR: {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"LINT OK: {len(rows)} rows; {len(warnings)} warnings")
    return 0


def main() -> int:
    """CLI entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runtime_root", required=True)
    args = ap.parse_args()
    return main_with_args(Path(args.csv), Path(args.runtime_root))


if __name__ == "__main__":
    sys.exit(main())
