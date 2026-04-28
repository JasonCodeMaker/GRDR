import argparse
import csv
import json
from pathlib import Path


DATASETS = ("msrvtt", "actnet", "didemo", "lsmdc")
SETTINGS = (1, 2)
INDEX_TYPES = ("hnsw", "ivf")
COMPACT_FIELDS = [
    "dataset",
    "setting",
    "index_type",
    "k",
    "stage1_recall_at_k",
    "stage1_r1",
    "stage1_r5",
    "stage1_r10",
    "stage1_medr",
    "stage1_meanr",
    "stage2_r1",
    "stage2_r5",
    "stage2_r10",
    "stage2_medr",
    "stage2_meanr",
]
K_BY_PAIR = {
    ("msrvtt", 1): 100, ("msrvtt", 2): 100,
    ("actnet", 1): 100, ("actnet", 2): 100,
    ("didemo", 1): 100, ("didemo", 2): 100,
    ("lsmdc", 1): 100, ("lsmdc", 2): 100,
}


def load_stage1_results(stage1_root: Path, index_type: str, dataset: str, setting: int):
    result_path = stage1_root / index_type / f"{dataset}_setting{setting}_pqt_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing Stage 1 summary: {result_path}")
    with result_path.open() as f:
        return json.load(f)


def load_candidate_json(candidate_dir: Path, dataset: str, index_type: str, k: int, setting: int):
    path = candidate_dir / f"{dataset}_ann_{index_type}_{k}_candidates_t{setting}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing candidate JSON: {path}")
    with path.open() as f:
        return path, json.load(f)


def load_stage2_results(
    stage2_dir: Path,
    dataset: str,
    index_type: str,
    k: int,
    setting: int,
    candidate_path: Path,
):
    path = stage2_dir / dataset / f"ann_{index_type}_{k}_candidates_t{setting}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage 2 CSV: {path}")
    if path.stat().st_mtime < candidate_path.stat().st_mtime:
        raise FileNotFoundError(
            f"Stage 2 CSV is older than candidate JSON, likely stale from a previous run: {path}"
        )
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected one row in Stage 2 CSV, got {len(rows)}: {path}")
    return rows[0]


def as_float(value):
    if value is None or value == "":
        return ""
    return float(value)


def format_metric(value):
    if value == "":
        return ""
    return f"{float(value):.2f}"


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize ANN baseline Stage 1 and Stage 2 metrics")
    parser.add_argument("--stage1_root", type=Path, required=True)
    parser.add_argument("--candidate_dir", type=Path, required=True)
    parser.add_argument("--stage2_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args()

    rows = []
    for dataset in DATASETS:
        for setting in SETTINGS:
            k = K_BY_PAIR[(dataset, setting)]
            for index_type in INDEX_TYPES:
                stage1 = load_stage1_results(args.stage1_root, index_type, dataset, setting)
                stage1_metrics = stage1["results"][index_type]
                candidate_path, candidate_json = load_candidate_json(
                    args.candidate_dir, dataset, index_type, k, setting
                )
                stage2_metrics = load_stage2_results(
                    args.stage2_dir, dataset, index_type, k, setting, candidate_path
                )
                row = {
                    "dataset": dataset,
                    "setting": setting,
                    "index_type": index_type,
                    "k": k,
                    "pool_size": candidate_json["metadata"]["pool_size"],
                    "num_queries": candidate_json["metrics"]["total_queries"],
                    "stage1_recall_at_k": format_metric(candidate_json["metrics"][f"Recall@{k}"]),
                    "stage1_r1": format_metric(as_float(stage1_metrics["R@1"])),
                    "stage1_r5": format_metric(as_float(stage1_metrics["R@5"])),
                    "stage1_r10": format_metric(as_float(stage1_metrics["R@10"])),
                    "stage1_medr": format_metric(as_float(stage1_metrics["MedR"])),
                    "stage1_meanr": format_metric(as_float(stage1_metrics["MeanR"])),
                    "stage1_text_encode_ms_mean": format_metric(as_float(stage1_metrics.get("encode_time_per_query_ms_mean"))),
                    "stage1_text_encode_ms_std": format_metric(as_float(stage1_metrics.get("encode_time_per_query_ms_std"))),
                    "stage1_index_load_ms_mean": format_metric(as_float(stage1_metrics.get("index_load_time_per_query_ms_mean"))),
                    "stage1_index_load_ms_std": format_metric(as_float(stage1_metrics.get("index_load_time_per_query_ms_std"))),
                    "stage1_search_ms_mean": format_metric(as_float(stage1_metrics.get("search_time_per_query_ms_mean"))),
                    "stage1_search_ms_std": format_metric(as_float(stage1_metrics.get("search_time_per_query_ms_std"))),
                    "stage1_online_total_ms_mean": format_metric(as_float(stage1_metrics.get("online_time_per_query_ms_mean"))),
                    "stage1_online_total_ms_std": format_metric(as_float(stage1_metrics.get("online_time_per_query_ms_std"))),
                    "stage1_offline_video_load_s": format_metric(as_float(stage1.get("video_embedding_load_time_s"))),
                    "stage1_build_s": format_metric(as_float(stage1_metrics["build_time_s"])),
                    "stage1_offline_total_s": format_metric(
                        as_float(stage1.get("video_embedding_load_time_s", 0.0)) +
                        as_float(stage1_metrics["build_time_s"])
                    ),
                    "stage2_r1": format_metric(as_float(stage2_metrics["R@1"])),
                    "stage2_r5": format_metric(as_float(stage2_metrics["R@5"])),
                    "stage2_r10": format_metric(as_float(stage2_metrics["R@10"])),
                    "stage2_medr": format_metric(as_float(stage2_metrics["MedR"])),
                    "stage2_meanr": format_metric(as_float(stage2_metrics["MeanR"])),
                }
                rows.append(row)

    fieldnames = list(rows[0].keys())
    write_csv(args.output, rows, fieldnames)

    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.compact_output is not None:
        compact_rows = [{field: row[field] for field in COMPACT_FIELDS} for row in rows]
        write_csv(args.compact_output, compact_rows, COMPACT_FIELDS)
        print(f"Wrote {len(compact_rows)} rows to {args.compact_output}")


if __name__ == "__main__":
    main()
