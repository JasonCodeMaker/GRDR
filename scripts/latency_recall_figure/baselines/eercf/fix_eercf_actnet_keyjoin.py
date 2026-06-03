#!/usr/bin/env python3
"""One-off repair: key-join EERCF sim-matrix rows to the package query set.

The default import (import_eercf_matrix.py) assumes matrix row i == TSV query i.
This holds for msrvtt/didemo/lsmdc but NOT activity (matrix=4917 rows from
anet_ret_val_1.json vs TSV=4918 from a different ACTNET enumeration). This script
recovers the matrix's true row/column order from the EERCF test JSON, matches each
TSV query to its matrix row by ground_truth_video_id, and emits candidate JSONs.

Self-validates against an already-correct dataset (didemo) before trusting ACTNET:
regenerates didemo candidates from the JSON-native order and diffs against the
existing import output. Identical -> ordering logic is correct.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import import_eercf_matrix as imp  # reuse helpers

# EERCF test JSON per datatype (the file that defines matrix row + test-col order).
TEST_JSON = {
    "activity": "ACTNET/video_retreival_caption/anet_ret_val_1.json",
    "didemo": "DIDEMO/video_retreival_caption/didemo_ret_test.json",
}


def matrix_row_video_ids(datatype: str, eercf_data_root: Path) -> list[str]:
    """Recover the matrix's row/test-col video order from the EERCF test JSON."""
    path = eercf_data_root / TEST_JSON[datatype]
    data = json.load(open(path, encoding="utf-8"))
    # ActivityNet/DiDeMo are one-to-one: one entry per video, in dataloader order.
    return [imp.clean_train_id(datatype, item["video"]) for item in data]


def build_candidates(datatype, setting, matrix, row_video_ids, pool_ids, tsv_rows):
    """Match each TSV row to its matrix row by gt_video_id; emit top-100 candidates."""
    row_index = {vid: i for i, vid in enumerate(row_video_ids)}
    pool_size = len(pool_ids)
    results, matched, unmatched = [], 0, 0
    for q in tsv_rows:
        gt = q["ground_truth_video_id"]
        mi = row_index.get(gt)
        if mi is None:
            unmatched += 1
            continue
        ranked = imp.top_indices_desc(np.asarray(matrix[mi, :pool_size]), 100)
        results.append({
            "query_text": q["query_text"],
            "ground_truth_video_id": gt,
            "candidates": [pool_ids[int(i)] for i in ranked],
            "scores": [float(matrix[mi, int(i)]) for i in ranked],
            "num_candidates": int(len(ranked)),
        })
        matched += 1
    return results, matched, unmatched


def prepare(datatype, setting, pkg_root, query_set_root, eercf_data_root, video_cache_root):
    """Load matrix + reconstruct row/col order + filtered TSV rows for one datatype."""
    matrix_path = imp.matrix_path_for(pkg_root, datatype, setting)
    matrix = np.load(matrix_path, mmap_mode="r")
    row_video_ids = matrix_row_video_ids(datatype, eercf_data_root)
    # Columns: unique test videos (JSON order) + train videos (import's load_train_ids).
    test_col_ids = row_video_ids
    train_col_ids = imp.load_train_ids(datatype, eercf_data_root, set(test_col_ids))
    pool_ids = test_col_ids + train_col_ids if setting == 2 else test_col_ids
    # TSV query rows, filtered by cache existence (same filter as the import).
    dsl = imp.DATATYPE_TO_DSL[datatype]
    rows = imp.load_query_rows(query_set_root, dsl, setting)
    upper, lower = imp.CACHE_SUB[datatype]
    rows = [
        r for r in rows
        if (video_cache_root / upper / lower
            / f"{imp._test_cache_key(datatype, r['ground_truth_video_id'])}.npz").exists()
    ]
    return matrix, matrix_path, row_video_ids, pool_ids, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg-root", type=Path, default=imp.DEFAULT_PKG_ROOT)
    ap.add_argument("--candidate-root", type=Path, default=None)
    ap.add_argument("--query-set-root", type=Path, default=imp.DEFAULT_QUERY_SET_ROOT)
    ap.add_argument("--eercf-data-root", type=Path, default=imp.DEFAULT_EERCF_DATA_ROOT)
    ap.add_argument("--video-cache-root", type=Path, default=imp.DEFAULT_VIDEO_CACHE_ROOT)
    ap.add_argument("--init-model", type=str, default=str(imp.DEFAULT_INIT_MODEL))
    ap.add_argument("--datatype", default="activity")
    ap.add_argument("--setting", type=int, default=2)
    ap.add_argument("--ops", type=int, nargs="+", default=[1, 10, 25, 50])
    ap.add_argument("--validate-against", default="didemo",
                    help="working datatype to diff against existing import output")
    args = ap.parse_args()
    cand_root = args.candidate_root or (args.pkg_root / "candidates_a")

    # --- Validation: reconstruct the working dataset, diff vs existing import JSON. ---
    if args.validate_against:
        vd = args.validate_against
        matrix, _, rows_vid, pool_ids, tsv = prepare(
            vd, args.setting, args.pkg_root, args.query_set_root,
            args.eercf_data_root, args.video_cache_root)
        recon, matched, unmatched = build_candidates(
            vd, args.setting, matrix, rows_vid, pool_ids, tsv)
        dsl = imp.DATATYPE_TO_DSL[vd]
        existing_path = next(
            iter((cand_root / "eercf" / dsl).glob(f"{dsl}_eercf_*_candidates_t{args.setting}.json")),
            None)
        if existing_path is None:
            print(f"[validate] no existing {vd} JSON to diff; aborting", file=sys.stderr)
            return 2
        existing = json.load(open(existing_path))["results"]
        # Compare candidate id lists per query (order-sensitive).
        recon_map = {r["ground_truth_video_id"]: r["candidates"] for r in recon}
        mismatches = sum(
            1 for r in existing
            if recon_map.get(r["ground_truth_video_id"]) != r["candidates"]
        )
        print(f"[validate] {vd}: existing={len(existing)} recon_matched={matched} "
              f"unmatched={unmatched} candidate-list mismatches={mismatches}", file=sys.stderr)
        if mismatches != 0:
            print(f"[validate] FAIL: {mismatches} mismatches; reconstruction unsafe, aborting",
                  file=sys.stderr)
            return 1
        print("[validate] PASS: reconstruction reproduces existing import exactly", file=sys.stderr)

    # --- Repair: emit the target datatype candidates. ---
    matrix, matrix_path, rows_vid, pool_ids, tsv = prepare(
        args.datatype, args.setting, args.pkg_root, args.query_set_root,
        args.eercf_data_root, args.video_cache_root)
    results, matched, unmatched = build_candidates(
        args.datatype, args.setting, matrix, rows_vid, pool_ids, tsv)
    print(f"[repair] {args.datatype}: matched={matched} unmatched={unmatched} "
          f"pool={len(pool_ids)} matrix={tuple(matrix.shape)}", file=sys.stderr)
    avg = sum(r["num_candidates"] for r in results) / max(1, len(results))
    for op in args.ops:
        payload = {
            "metadata": {
                "dataset": imp.DATATYPE_TO_DSL[args.datatype],
                "model_name": "EERCF-CLIP4Clip", "num_candidates": 100,
                "setting": int(args.setting), "checkpoint": args.init_model,
                "matrix_path": str(matrix_path), "keyjoin_repair": True,
            },
            "metrics": {
                "total_queries": len(results), "avg_candidates_per_query": avg,
                "rerantopk": int(op), "pool_size": len(pool_ids),
                "pool_scope": "train+test" if args.setting == 2 else "test-only",
            },
            "results": results,
        }
        out = imp.write_candidate_json(cand_root, args.datatype, args.setting, op, payload)
        print(f"[repair] wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
