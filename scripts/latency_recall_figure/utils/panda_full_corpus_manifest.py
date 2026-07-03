#!/usr/bin/env python
"""Classify Panda full-corpus figure rows against the package TDD gates."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_QUERY_COUNT = 5694
EXPECTED_POOL_SIZE = 2156234
EXPECTED_GRID = {
    "grdr_ref": [20, 50, 100, 200, 300],
    "tiger": [20, 50, 100, 200, 300],
    "avg": [20, 50, 100, 200, 300],
    "t2vindexer": [20, 50, 100, 200, 300],
    "eercf": [50],
    "hnsw": [20, 40, 100, 200],
    "ivf": [20, 40, 100, 200],
    "ivfpq": [20, 40, 100, 200],
    "opq": [20, 40, 100, 200],
}
ANN_METHODS = {"hnsw", "ivf", "ivfpq", "opq"}
GR_METHODS = {"grdr_ref", "tiger", "avg"}
DIAGNOSTIC_EFFECTS = {"OOM", "unsupported"}
CLASSIFIED_CLASSES = {"valid", "diagnostic_only"}
RESULT_COLUMNS = [
    "row_id", "exp_id", "metric", "value", "unit", "split", "baseline",
    "verdict", "validity", "source_artifact", "source_mtime", "extractor", "extracted_at",
]


def expected_cells() -> list[tuple[str, int]]:
    return [(method, op) for method, ops in EXPECTED_GRID.items() for op in ops]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _to_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    num = _to_float(value)
    return int(num) if num is not None else None


def _resolve(runtime_root: Path, rel_or_abs: str) -> Path | None:
    if not rel_or_abs:
        return None
    path = Path(rel_or_abs)
    if path.is_absolute():
        return path
    return (runtime_root / path).resolve()


def _repo_candidate_path(repo_root: Path | None, method: str, op: int) -> Path | None:
    if repo_root is None:
        return None
    if method == "grdr_ref":
        return repo_root / "candidates" / "GRDR" / "panda" / f"panda_t2_{op}_candidates.json"
    if method in {"tiger", "avg"}:
        return repo_root / "candidates" / "baselines" / method / "panda" / f"panda_{method}_{op}_candidates_t2.json"
    if method == "t2vindexer":
        return repo_root / "candidates" / "baselines" / method / "panda" / f"panda_t2_{op}_candidates.json"
    if method == "eercf":
        return repo_root / "candidates" / "baselines" / method / "panda" / f"panda_eercf_{op}_candidates_t2.json"
    if method in ANN_METHODS:
        return repo_root / "candidates" / "baselines" / method / "panda" / f"panda_ann_{method}_{op}_candidates_t2.json"
    return None


def _candidate_json(row: dict[str, str], runtime_root: Path, repo_root: Path | None, method: str, op: int) -> tuple[Path | None, dict[str, Any] | None]:
    paths: list[Path] = []
    if method in ANN_METHODS:
        resolved = _resolve(runtime_root, row.get("rerank_source_path", ""))
        if resolved is not None:
            paths.append(resolved)
    fallback = _repo_candidate_path(repo_root, method, op)
    if fallback is not None:
        paths.append(fallback)
    for path in paths:
        payload = _load_json(path)
        if payload is not None:
            return path, payload
    return (paths[0] if paths else None), None


def _latency_json(row: dict[str, str], runtime_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    path = _resolve(runtime_root, row.get("stage1_source_path", ""))
    if path is None:
        return None, None
    return path, _load_json(path)


def _classify_recall(row: dict[str, str] | None, runtime_root: Path, repo_root: Path | None, method: str, op: int) -> tuple[str, list[str], str | None]:
    if row is None:
        return "missing", ["row_missing"], None
    eff = str(row.get("effectiveness_validity") or "missing")
    if eff in DIAGNOSTIC_EFFECTS:
        return "diagnostic_only", [f"effectiveness_validity={eff}"], None
    if eff != "full_test_set":
        return ("missing" if eff == "missing" else "failed"), [f"effectiveness_validity={eff}"], None

    reasons: list[str] = []
    if _to_int(row.get("n_queries")) != EXPECTED_QUERY_COUNT:
        reasons.append(f"n_queries!={EXPECTED_QUERY_COUNT}")
    for col in ("CanHit@20", "CanHit@50", "CanHit@100"):
        if _to_float(row.get(col)) is None:
            reasons.append(f"{col}_missing")

    cand_path, cand = _candidate_json(row, runtime_root, repo_root, method, op)
    cand_meta = (cand or {}).get("metadata") or {}
    cand_metrics = (cand or {}).get("metrics") or {}
    if cand is not None:
        if _to_int(cand_metrics.get("total_queries")) not in (None, EXPECTED_QUERY_COUNT):
            reasons.append(f"candidate_total_queries!={EXPECTED_QUERY_COUNT}")
        if method in ANN_METHODS:
            if _to_int(cand_meta.get("pool_size")) != EXPECTED_POOL_SIZE:
                reasons.append(f"ann_pool_size!={EXPECTED_POOL_SIZE}")
            if cand_meta.get("feature_backbone") != "xpool_clip":
                reasons.append("ann_feature_backbone_not_xpool_clip")
        if method in GR_METHODS:
            cap = _to_int(cand_meta.get("candidate_handoff_cap") or cand_metrics.get("candidate_handoff_cap"))
            if cap != 3 * op:
                reasons.append(f"gr_cap!={3 * op}")
    elif method in ANN_METHODS:
        reasons.append("ann_candidate_witness_missing")

    return ("fix_first" if reasons else "valid"), reasons, str(cand_path) if cand_path else None


def _classify_latency(row: dict[str, str] | None, runtime_root: Path, method: str, op: int) -> tuple[str, list[str], str | None]:
    if row is None:
        return "missing", ["row_missing"], None
    eff = str(row.get("effectiveness_validity") or "missing")
    lat = str(row.get("latency_validity") or "missing")
    if eff in DIAGNOSTIC_EFFECTS and lat == "missing":
        return "diagnostic_only", [f"effectiveness_validity={eff}; latency intentionally blank"], None
    if lat == "missing":
        return "missing", ["latency_validity=missing"], None

    reasons: list[str] = []
    stage1 = _to_float(row.get("stage1_latency_ms"))
    stage2 = _to_float(row.get("stage2_latency_ms"))
    total = _to_float(row.get("total_latency_ms"))
    if stage1 is None or stage1 <= 0:
        reasons.append("stage1_latency_ms_missing_or_nonpositive")
    if method in ANN_METHODS:
        if stage2 is not None:
            reasons.append("ann_stage2_latency_present")
        if total is not None and stage1 is not None and abs(total - stage1) > 1e-6:
            reasons.append("ann_total_latency_not_stage1")
        lat_path, lat_json = _latency_json(row, runtime_root)
        meta = (lat_json or {}).get("metadata") or {}
        if lat_json is not None:
            if _to_int(meta.get("pool_size")) not in (None, EXPECTED_POOL_SIZE):
                reasons.append(f"ann_latency_pool_size!={EXPECTED_POOL_SIZE}")
            if meta.get("feature_backbone") not in (None, "xpool_clip"):
                reasons.append("ann_latency_feature_backbone_not_xpool_clip")
        elif lat_path is not None:
            reasons.append("latency_witness_missing")
    return ("fix_first" if reasons else "valid"), reasons, row.get("stage1_source_path") or None


def classify_rows(csv_path: Path, runtime_root: Path, expected: list[tuple[str, int]] | None = None,
                  repo_root: Path | None = None) -> dict[str, Any]:
    runtime_root = Path(runtime_root).resolve()
    repo_root = Path(repo_root).resolve() if repo_root else None
    expected = expected or expected_cells()
    with Path(csv_path).open() as f:
        rows = list(csv.DictReader(f))
    panda_rows = [r for r in rows if r.get("dataset") == "PANDA" and str(r.get("setting")) == "2"]
    row_map: dict[tuple[str, int], dict[str, str]] = {}
    duplicate_cells: list[str] = []
    for row in panda_rows:
        key = (row.get("method", ""), _to_int(row.get("op_point_value")) or -1)
        if key in row_map:
            duplicate_cells.append(f"{key[0]}:{key[1]}")
        row_map[key] = row

    cells: list[dict[str, Any]] = []
    for method, op in expected:
        row = row_map.get((method, op))
        recall_class, recall_reasons, candidate_path = _classify_recall(row, runtime_root, repo_root, method, op)
        latency_class, latency_reasons, latency_path = _classify_latency(row, runtime_root, method, op)
        cells.append({
            "method": method,
            "op": op,
            "recall_class": recall_class,
            "recall_reasons": recall_reasons,
            "latency_class": latency_class,
            "latency_reasons": latency_reasons,
            "candidate_source_path": candidate_path,
            "latency_source_path": latency_path,
        })

    recall_count = sum(1 for c in cells if c["recall_class"] in CLASSIFIED_CLASSES)
    latency_count = sum(1 for c in cells if c["latency_class"] in CLASSIFIED_CLASSES)
    return {
        "csv": str(csv_path),
        "runtime_root": str(runtime_root),
        "repo_root": str(repo_root) if repo_root else None,
        "expected_cells": len(expected),
        "panda_rows": len(panda_rows),
        "duplicate_cells": duplicate_cells,
        "recall_cells_classified": recall_count,
        "latency_cells_classified": latency_count,
        "cells": cells,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _result_row(exp_id: str, metric: str, value: Any, unit: str, split: str, baseline: str,
                verdict: str, validity: str, source_artifact: str) -> dict[str, str]:
    now = datetime.now(timezone.utc).isoformat()
    mtime = ""
    path = Path(source_artifact)
    if path.exists():
        mtime = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    return {
        "row_id": f"{exp_id}_gate",
        "exp_id": exp_id,
        "metric": metric,
        "value": str(value),
        "unit": unit,
        "split": split,
        "baseline": baseline,
        "verdict": verdict,
        "validity": validity,
        "source_artifact": source_artifact,
        "source_mtime": mtime,
        "extractor": "panda_full_corpus_manifest.py",
        "extracted_at": now,
    }


def _write_result_table(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--repo-root")
    parser.add_argument("--out-root", default="outputs/2026-06-22-panda-full-corpus-figure")
    parser.add_argument("--result-table-dir")
    args = parser.parse_args()

    summary = classify_rows(Path(args.csv), Path(args.runtime_root), repo_root=Path(args.repo_root) if args.repo_root else None)
    out_root = Path(args.out_root)
    recall_manifest = out_root / "P2" / "recall_manifest.json"
    latency_manifest = out_root / "P3" / "latency_manifest.json"
    aggregate_manifest = out_root / "P4" / "aggregate_manifest.json"
    _write_json(recall_manifest, {
        "gate": "recall_cells_classified == 37",
        "recall_cells_classified": summary["recall_cells_classified"],
        "expected_cells": summary["expected_cells"],
        "cells": summary["cells"],
    })
    _write_json(latency_manifest, {
        "gate": "latency_cells_classified == 37",
        "latency_cells_classified": summary["latency_cells_classified"],
        "expected_cells": summary["expected_cells"],
        "cells": summary["cells"],
    })
    _write_json(aggregate_manifest, {
        "gate": "panda_rows == 37",
        "panda_rows": summary["panda_rows"],
        "expected_cells": summary["expected_cells"],
        "duplicate_cells": summary["duplicate_cells"],
        "csv": summary["csv"],
    })

    if args.result_table_dir:
        tables = Path(args.result_table_dir)
        p2_pass = summary["recall_cells_classified"] == summary["expected_cells"]
        p3_pass = summary["latency_cells_classified"] == summary["expected_cells"]
        p4_pass = summary["panda_rows"] == summary["expected_cells"] and not summary["duplicate_cells"]
        _write_result_table(tables / "result_table_P2.csv", _result_row(
            "P2", "recall_cells_classified == 37", summary["recall_cells_classified"], "cells",
            "PANDA setting=2", "Panda full-corpus recall evidence classification",
            "PASS" if p2_pass else "INCONCLUSIVE", "VALID" if p2_pass else "UNMEASURED", str(recall_manifest),
        ))
        _write_result_table(tables / "result_table_P3.csv", _result_row(
            "P3", "latency_cells_classified == 37", summary["latency_cells_classified"], "cells",
            "PANDA setting=2", "Panda Stage-1 latency evidence classification",
            "PASS" if p3_pass else "INCONCLUSIVE", "VALID" if p3_pass else "UNMEASURED", str(latency_manifest),
        ))
        _write_result_table(tables / "result_table_P4.csv", _result_row(
            "P4", "panda_rows == 37", summary["panda_rows"], "rows",
            "PANDA setting=2", "Canonical figure_data.csv Panda row block",
            "PASS" if p4_pass else "INCONCLUSIVE", "VALID" if p4_pass else "UNMEASURED", str(aggregate_manifest),
        ))

    print(json.dumps({
        "recall_cells_classified": summary["recall_cells_classified"],
        "latency_cells_classified": summary["latency_cells_classified"],
        "panda_rows": summary["panda_rows"],
        "expected_cells": summary["expected_cells"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
