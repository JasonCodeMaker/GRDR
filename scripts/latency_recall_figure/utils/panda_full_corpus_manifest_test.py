#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from panda_full_corpus_manifest import classify_rows, expected_cells


FIELDS = [
    "dataset", "setting", "method",
    "op_point_label", "op_point_knob", "op_point_value",
    "stage1_latency_ms", "stage2_latency_ms", "total_latency_ms",
    "CanHit@20", "CanHit@50", "CanHit@100",
    "R@1", "R@5", "R@10",
    "n_queries", "avg_candidates_per_query",
    "latency_validity", "effectiveness_validity",
    "stage1_source_path", "rerank_source_path",
]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def base_row(method: str, op: int, *, effect: str = "full_test_set", latency: str = "full_subset") -> dict[str, object]:
    return {
        "dataset": "PANDA",
        "setting": 2,
        "method": method,
        "op_point_label": f"budget={op}",
        "op_point_knob": "budget",
        "op_point_value": op,
        "stage1_latency_ms": 10.0,
        "stage2_latency_ms": "",
        "total_latency_ms": 10.0,
        "CanHit@20": 1.0,
        "CanHit@50": 2.0,
        "CanHit@100": 3.0,
        "R@1": 0.1,
        "R@5": 0.2,
        "R@10": 0.3,
        "n_queries": 5694,
        "avg_candidates_per_query": op,
        "latency_validity": latency,
        "effectiveness_validity": effect,
        "stage1_source_path": f"recall-latency/{method}/panda/panda_t2_{op}_latency.json",
        "rerank_source_path": f"candidates/{method}_{op}.json",
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        csv_path = root / "figure_data.csv"
        rows = [
            base_row("hnsw", 200),
            {**base_row("t2vindexer", 100, effect="OOM", latency="missing"), "stage1_latency_ms": "", "total_latency_ms": ""},
            {**base_row("avg", 300, latency="missing"), "stage1_latency_ms": "", "total_latency_ms": ""},
            {**base_row("ivf", 100), "stage2_latency_ms": 4.0, "total_latency_ms": 14.0},
        ]
        write_csv(csv_path, rows)

        (root / "recall-latency/hnsw/panda").mkdir(parents=True)
        (root / "recall-latency/hnsw/panda/panda_t2_200_latency.json").write_text(json.dumps({
            "metadata": {
                "dataset": "panda",
                "setting": 2,
                "pool_size": 2156234,
                "feature_backbone": "xpool_clip",
                "stage1_latency_ms": {"validity": "full_subset"},
            }
        }))
        (root / "candidates").mkdir(parents=True)
        (root / "candidates/hnsw_200.json").write_text(json.dumps({
            "metadata": {
                "dataset": "panda",
                "setting": 2,
                "pool_size": 2156234,
                "feature_backbone": "xpool_clip",
            },
            "metrics": {"total_queries": 5694},
            "results": [{"ground_truth_video_id": "v1", "candidates": ["v1"]}],
        }))

        summary = classify_rows(csv_path, root, expected=expected_cells())
        by_method = {(c["method"], c["op"]): c for c in summary["cells"]}

        assert by_method[("hnsw", 200)]["recall_class"] == "valid"
        assert by_method[("hnsw", 200)]["latency_class"] == "valid"
        assert by_method[("t2vindexer", 100)]["recall_class"] == "diagnostic_only"
        assert by_method[("t2vindexer", 100)]["latency_class"] == "diagnostic_only"
        assert by_method[("avg", 300)]["recall_class"] == "valid"
        assert by_method[("avg", 300)]["latency_class"] == "missing"
        assert by_method[("ivf", 100)]["latency_class"] == "fix_first"
        assert len(expected_cells()) == 37
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
