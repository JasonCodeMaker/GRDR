#!/usr/bin/env python3
"""Slice EERCF sim-matrices into package-uniform Pass-A candidate JSONs.

Adapted from the prior panda baselines importer (vendored; native paths).
Local changes:
  - default runtime root points at this latency-recall package
  - output candidate root is configurable, so Pass-A writes under candidates_a/
  - output filename can carry the EERCF operating-point value
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


LSMDC_TIMESTAMP_RE = re.compile(r"_(\d+\.\d+\.\d+\.\d+-\d+\.\d+\.\d+\.\d+)$")


DATASETS = ("msrvtt", "activity", "didemo", "lsmdc")
SETTINGS = (1, 2)

# Map EERCF lowercase datatypes to the package-uniform JSON name (dsl).
DATATYPE_TO_DSL = {
    "msrvtt": "msrvtt",
    "activity": "actnet",
    "didemo": "didemo",
    "lsmdc": "lsmdc",
}

DEFAULT_PKG_ROOT = Path(
    "/home/uqzzha35/Project/SemanticID/GRDR/output/evaluation_results/figures"
)
DEFAULT_INPUT_ROOT = Path(
    "/home/uqzzha35/Project/SemanticID/GRDR/output/checkpoints/Baseline"
)
DEFAULT_QUERY_SET_ROOT = Path(
    "/home/uqzzha35/Project/SemanticID/GRDR/output/evaluation_results/figures_panda/query_sets"
)
DEFAULT_EERCF_DATA_ROOT = Path("/home/uqzzha35/Project/SemanticID/EERCF/data")
DEFAULT_VIDEO_CACHE_ROOT = DEFAULT_INPUT_ROOT / "eercf" / "panda" / "cached_video_features_p3d"
DEFAULT_INIT_MODEL = (
    DEFAULT_INPUT_ROOT / "eercf" / "panda" / "pytorch_model.bin.best.0"
)

CACHE_SUB = {
    "msrvtt":   ("MSRVTT", "msrvtt"),
    "activity": ("ACTNET", "activity"),
    "didemo":   ("DIDEMO", "didemo"),
    "lsmdc":    ("LSMDC",  "lsmdc"),
}


def clean_train_id(datatype: str, value: str) -> str:
    """Normalize a raw train-set video id to the cached .npz filename stem."""
    text = str(value)
    if datatype == "lsmdc":
        text = text.replace("/", "_").replace(".avi", "").replace(".mp4", "")
    elif datatype in {"activity", "didemo", "msrvtt"}:
        text = text.replace(".mp4", "").replace(".avi", "")
    return text


def load_query_rows(query_set_root: Path, dsl: str, setting: int) -> list[dict[str, str]]:
    """Read the package-shared query TSV (single source of truth for query+GT order)."""
    path = query_set_root / f"{dsl}_setting{setting}_queries.tsv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise ValueError(f"empty query manifest: {path}")
    return rows


def load_train_ids(datatype: str, eercf_data_root: Path, test_ids: set[str]) -> list[str]:
    """LSMDC: sort by format-B (EERCF cache key), emit format-C (X-Pool canonical)."""
    if datatype == "msrvtt":
        path = eercf_data_root / "MSRVTT/raw/MSRVTT_train.9k.csv"
        with path.open("r", encoding="utf-8", newline="") as fh:
            ids = {row["video_id"] for row in csv.DictReader(fh)}
        return sorted(vid for vid in ids if vid and vid not in test_ids)
    elif datatype == "activity":
        path = eercf_data_root / "ACTNET/video_retreival_caption/anet_ret_train.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        ids = {clean_train_id(datatype, item.get("video", "")) for item in data}
        return sorted(vid for vid in ids if vid and vid not in test_ids)
    elif datatype == "didemo":
        path = eercf_data_root / "DIDEMO/video_retreival_caption/didemo_ret_train.json"
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        ids = {clean_train_id(datatype, item.get("video", "")) for item in data}
        return sorted(vid for vid in ids if vid and vid not in test_ids)
    elif datatype == "lsmdc":
        train_path = eercf_data_root / "LSMDC/video_retreival_caption/lsmdc_ret_train.json"
        test_path = eercf_data_root / "LSMDC/video_retreival_caption/lsmdc_ret_test_1000.json"
        with train_path.open("r", encoding="utf-8") as fh:
            train_data = json.load(fh)
        with test_path.open("r", encoding="utf-8") as fh:
            test_data = json.load(fh)
        test_format_b = {
            str(item.get("video", "")).replace("/", "_").replace(".avi", "")
            for item in test_data
        }
        b_to_c: dict[str, str] = {}
        for item in train_data:
            raw = str(item.get("video", ""))
            if not raw:
                continue
            b = raw.replace("/", "_").replace(".avi", "")
            if b in test_format_b:
                continue
            c = raw.rsplit("/", 1)[-1].replace(".avi", "")
            b_to_c[b] = c
        return [b_to_c[b] for b in sorted(b_to_c.keys())]
    else:
        raise ValueError(f"unknown datatype: {datatype}")


def matrix_path_for(pkg_root: Path, datatype: str, setting: int) -> Path:
    """Resolve EERCF sim-matrix filename per setting (different basenames)."""
    name = "expanded_pool_sim_matrix.npy" if setting == 2 else "sim_matrix.npy"
    return pkg_root / "matrices" / "eercf" / datatype / f"setting{setting}" / name


def top_indices_desc(row: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k values in descending order (stable for ties via argsort)."""
    k = int(min(k, row.shape[0]))
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.argpartition(-row, k - 1)[:k]
    return idx[np.argsort(-row[idx])]


def build_candidate_payload(
    datatype: str,
    setting: int,
    rerantopk: int,
    init_model: str,
    matrix_path: Path,
    matrix: np.ndarray,
    rows: list[dict[str, str]],
    pool_ids: list[str],
) -> dict[str, Any]:
    """Assemble the package-uniform candidate JSON for one cell."""
    print(
        f"[shape] {datatype} setting{setting}: matrix={tuple(matrix.shape)}, "
        f"pool_size={len(pool_ids)}",
        file=sys.stderr,
    )
    if matrix.shape[0] != len(rows):
        raise ValueError(
            f"{datatype} setting{setting}: matrix rows {matrix.shape[0]} != "
            f"query rows {len(rows)}"
        )
    if matrix.shape[1] != len(pool_ids):
        raise ValueError(
            f"{datatype} setting{setting}: matrix shape {tuple(matrix.shape)} "
            f"cols != pool_size {len(pool_ids)}"
        )

    pool_size = len(pool_ids)
    results: list[dict[str, Any]] = []
    for q_idx, query in enumerate(rows):
        ranked = top_indices_desc(matrix[q_idx, :pool_size], 100)
        candidates = [pool_ids[int(i)] for i in ranked]
        scores = [float(matrix[q_idx, int(i)]) for i in ranked]
        results.append(
            {
                "query_text": query["query_text"],
                "ground_truth_video_id": query["ground_truth_video_id"],
                "candidates": candidates,
                "scores": scores,
                "num_candidates": len(candidates),
            }
        )

    avg_count = (
        float(sum(r["num_candidates"] for r in results)) / max(1, len(results))
    )
    pool_scope = "test-only" if setting == 1 else "train+test"

    return {
        "metadata": {
            "dataset": DATATYPE_TO_DSL[datatype],
            "model_name": "EERCF-CLIP4Clip",
            "num_candidates": 100,
            "setting": int(setting),
            "checkpoint": init_model,
            "matrix_path": str(matrix_path),
            "timestamp": time.strftime("%m%d%H%M"),
        },
        "metrics": {
            "total_queries": len(rows),
            "avg_candidates_per_query": avg_count,
            "rerantopk": int(rerantopk),
            "pool_size": pool_size,
            "pool_scope": pool_scope,
        },
        "results": results,
    }


def write_candidate_json(
    candidate_root: Path,
    datatype: str,
    setting: int,
    output_op: int | None,
    payload: dict[str, Any],
) -> Path:
    """Write the candidate JSON at the exact path sweep_pass_a expects."""
    dsl = DATATYPE_TO_DSL[datatype]
    out_dir = candidate_root / "eercf" / dsl
    out_dir.mkdir(parents=True, exist_ok=True)
    op_label = int(output_op) if output_op is not None else 100
    out_path = out_dir / f"{dsl}_eercf_{op_label}_candidates_t{setting}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _test_cache_key(datatype: str, test_pool_id: str) -> str:
    """Map TSV ground_truth_video_id (X-Pool canonical) to the .npz filename stem."""
    if datatype != "lsmdc":
        return test_pool_id
    # LSMDC: TSV is format-C (MOVIE_TIMESTAMP); cache is format-B (MOVIE_MOVIE_TIMESTAMP).
    m = LSMDC_TIMESTAMP_RE.search(test_pool_id)
    if not m:
        return test_pool_id
    prefix = test_pool_id[: m.start()]
    return f"{prefix}_{prefix}_{m.group(1)}"


def run_cell(
    pkg_root: Path,
    candidate_root: Path,
    query_set_root: Path,
    eercf_data_root: Path,
    datatype: str,
    setting: int,
    rerantopk: int,
    output_op: int | None,
    init_model: str,
    video_cache_root: Path = DEFAULT_VIDEO_CACHE_ROOT,
) -> tuple[Path, int, int, tuple[int, int]] | None:
    """Import one (datatype, setting) cell; return None if matrix missing."""
    matrix_path = matrix_path_for(pkg_root, datatype, setting)
    if not matrix_path.exists():
        print(f"[skip] missing matrix: {matrix_path}", file=sys.stderr)
        return None

    dsl = DATATYPE_TO_DSL[datatype]
    rows = load_query_rows(query_set_root, dsl, setting)
    test_ids = [row["ground_truth_video_id"] for row in rows]
    # Filter test rows whose video-level cache .npz is missing — matches A4 filter.
    upper, lower = CACHE_SUB[datatype]
    keep = [
        i for i, vid in enumerate(test_ids)
        if (video_cache_root / upper / lower / f"{_test_cache_key(datatype, vid)}.npz").exists()
    ]
    keep_mask = None
    if len(keep) != len(test_ids):
        dropped = len(test_ids) - len(keep)
        print(
            f"[filter] {datatype} setting{setting}: dropping {dropped} test rows "
            f"with missing video cache (TSV stale ids)",
            file=sys.stderr,
        )
        rows = [rows[i] for i in keep]
        test_ids = [test_ids[i] for i in keep]
        keep_mask = keep  # row indices in original full test set
    if setting == 1:
        pool_ids = test_ids
    else:
        pool_ids = test_ids + load_train_ids(datatype, eercf_data_root, set(test_ids))

    matrix = np.load(matrix_path, mmap_mode="r")
    # Align matrix to filtered queries: drop the same stale test indices from rows AND
    # from the leading test-cols block (cols layout in expanded_pool: test + train).
    if keep_mask is not None and setting == 2:
        n_test_full = matrix.shape[0]  # original test count before filter
        n_train = matrix.shape[1] - n_test_full
        col_keep = list(keep_mask) + list(range(n_test_full, n_test_full + n_train))
        matrix = np.asarray(matrix[np.ix_(keep_mask, col_keep)])
    elif keep_mask is not None:
        matrix = np.asarray(matrix[keep_mask, :])
    payload = build_candidate_payload(
        datatype, setting, rerantopk, init_model, matrix_path, matrix, rows, pool_ids,
    )
    out_path = write_candidate_json(candidate_root, datatype, setting, output_op, payload)
    return out_path, len(rows), len(pool_ids), tuple(int(x) for x in matrix.shape)


def self_test_lsmdc_normalization(eercf_data_root: Path) -> bool:
    """LSMDC train ids must emit format-C (single-prefix, no .avi)."""
    train_path = eercf_data_root / "LSMDC/video_retreival_caption/lsmdc_ret_train.json"
    if not train_path.exists():
        print(f"[self-test] SKIP lsmdc: missing {train_path}", file=sys.stderr)
        return True
    returned = load_train_ids("lsmdc", eercf_data_root, set())
    if not returned:
        print("[self-test] FAIL lsmdc: empty return", file=sys.stderr)
        return False
    if not (101055 - 100 <= len(returned) <= 101055 + 100):
        print(
            f"[self-test] FAIL lsmdc: unexpected count {len(returned)} "
            f"(expected ~101055 +/-100)",
            file=sys.stderr,
        )
        return False
    with train_path.open("r", encoding="utf-8") as fh:
        train_data = json.load(fh)
    # Recompute the expected first-3 from scratch (sort by format-B, emit format-C).
    pairs = []
    for item in train_data:
        raw = str(item.get("video", ""))
        if not raw:
            continue
        b = raw.replace("/", "_").replace(".avi", "")
        c = raw.rsplit("/", 1)[-1].replace(".avi", "")
        pairs.append((b, c))
    pairs.sort(key=lambda bc: bc[0])
    expected_first3 = [c for _, c in pairs[:3]]
    actual_first3 = returned[:3]
    if actual_first3 != expected_first3:
        print(
            f"[self-test] FAIL lsmdc: first3 mismatch\n"
            f"  expected={expected_first3}\n"
            f"  actual  ={actual_first3}",
            file=sys.stderr,
        )
        return False
    # Property check: no duplicated movie prefix in any emitted id.
    for vid in returned[:50]:
        head = vid.split("_00.", 1)[0] if "_00." in vid else vid
        # Duplicated prefix would look like "<HEAD>_<HEAD>_<timestamp>"
        if f"{head}_{head}_" in vid:
            print(
                f"[self-test] FAIL lsmdc: duplicated prefix in {vid!r}",
                file=sys.stderr,
            )
            return False
    print(f"[self-test] PASS lsmdc (n={len(returned)})")
    return True


def self_test(eercf_data_root: Path = DEFAULT_EERCF_DATA_ROOT) -> bool:
    """Synthetic 3-query x 5-pool matrix; verify top-2 sort + index bounds."""
    rng = np.random.default_rng(42)
    matrix = rng.random((3, 5), dtype=np.float32)
    pool_ids = [f"v{i}" for i in range(5)]
    rows = [
        {"query_text": f"q{i}", "ground_truth_video_id": pool_ids[i]} for i in range(3)
    ]
    payload = build_candidate_payload(
        datatype="msrvtt",
        setting=1,
        rerantopk=50,
        init_model="/dev/null",
        matrix_path=Path("/dev/null/synthetic"),
        matrix=matrix,
        rows=rows,
        pool_ids=pool_ids,
    )
    # Truncate top-K to 2 for the self-test by re-ranking explicitly.
    for q_idx, entry in enumerate(payload["results"]):
        ranked = top_indices_desc(matrix[q_idx], 2)
        entry["candidates"] = [pool_ids[int(i)] for i in ranked]
        entry["scores"] = [float(matrix[q_idx, int(i)]) for i in ranked]
        entry["num_candidates"] = 2

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "self_test.json"
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        reloaded = json.loads(tmp.read_text(encoding="utf-8"))

    for q_idx, entry in enumerate(reloaded["results"]):
        scores = entry["scores"]
        candidates = entry["candidates"]
        if len(scores) != 2 or len(candidates) != 2:
            print(f"[self-test] FAIL: q{q_idx} length != 2", file=sys.stderr)
            return False
        if not all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)):
            print(f"[self-test] FAIL: q{q_idx} not descending: {scores}", file=sys.stderr)
            return False
        for vid in candidates:
            if vid not in pool_ids:
                print(f"[self-test] FAIL: q{q_idx} out-of-bounds vid: {vid}", file=sys.stderr)
                return False
    print("[self-test] PASS synthetic")
    if not self_test_lsmdc_normalization(eercf_data_root):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import EERCF sim-matrices to package-uniform candidate JSONs."
    )
    parser.add_argument("--pkg-root", type=Path, default=DEFAULT_PKG_ROOT)
    parser.add_argument("--candidate-root", type=Path, default=None,
                        help="Pass-A candidate root; defaults to <pkg-root>/candidates_a.")
    parser.add_argument("--query-set-root", type=Path, default=DEFAULT_QUERY_SET_ROOT)
    parser.add_argument("--eercf-data-root", type=Path, default=DEFAULT_EERCF_DATA_ROOT)
    parser.add_argument("--video-cache-root", type=Path, default=DEFAULT_VIDEO_CACHE_ROOT,
                        help="Used to drop TSV test rows with missing cache .npz")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS),
                        help="EERCF datatypes (lowercase): msrvtt activity didemo lsmdc")
    parser.add_argument("--settings", nargs="*", type=int, default=list(SETTINGS))
    parser.add_argument("--rerantopk", type=int, default=50,
                        help="Metadata only; top-100 selection is unbounded by rerantopk.")
    parser.add_argument("--output-op", type=int, default=None,
                        help="Operating-point value to embed in the output filename.")
    parser.add_argument("--init-model", type=str, default=str(DEFAULT_INIT_MODEL),
                        help="Metadata only; recorded in candidate JSON metadata.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run synthetic-matrix unit check and exit.")
    args = parser.parse_args()

    if args.self_test:
        return 0 if self_test(args.eercf_data_root) else 1

    candidate_root = args.candidate_root or (args.pkg_root / "candidates_a")

    written = 0
    for datatype in args.datasets:
        if datatype not in DATATYPE_TO_DSL:
            print(f"[skip] unknown datatype: {datatype}", file=sys.stderr)
            continue
        for setting in args.settings:
            result = run_cell(
                pkg_root=args.pkg_root,
                candidate_root=candidate_root,
                query_set_root=args.query_set_root,
                eercf_data_root=args.eercf_data_root,
                datatype=datatype,
                setting=int(setting),
                rerantopk=args.rerantopk,
                output_op=args.output_op,
                init_model=args.init_model,
                video_cache_root=args.video_cache_root,
            )
            if result is None:
                continue
            out_path, n_queries, pool_size, shape = result
            print(
                f"[write] {out_path}  queries={n_queries}  "
                f"pool={pool_size}  matrix_shape={shape}"
            )
            written += 1

    if written == 0:
        print("[error] no cells written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
