import argparse
import csv
import json
from pathlib import Path


DATASETS = ("msrvtt", "actnet", "didemo", "lsmdc")
SETTINGS = (1, 2)
INDEX_TYPES = ("flat", "hnsw", "ivf")
K_BY_PAIR = {
    ("msrvtt", 1): 100, ("msrvtt", 2): 100,
    ("actnet", 1): 100, ("actnet", 2): 100,
    ("didemo", 1): 100, ("didemo", 2): 100,
    ("lsmdc", 1): 100, ("lsmdc", 2): 100,
}
COMPACT_FIELDS = [
    "dataset",
    "setting",
    "index_type",
    "k",
    "stage1_total_ms_mean",
    "stage2_total_ms_mean",
    "end_to_end_total_ms_mean",
]
STAGE2_FIELDS = [
    "dataset",
    "retrieval_mode",
    "candidate_file",
    "num_queries",
    "search_pool_size",
    "candidate_count_mean",
    "candidate_count_min",
    "candidate_count_max",
    "query_encode_ms_mean",
    "query_encode_ms_std",
    "video_load_ms_mean",
    "video_load_ms_std",
    "frame_pooling_ms_mean",
    "frame_pooling_ms_std",
    "similarity_compute_ms_mean",
    "similarity_compute_ms_std",
    "total_ms_mean",
    "total_ms_std",
]


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open() as f:
        return json.load(f)


def load_stage1_results(stage1_root: Path, dataset: str, setting: int, index_type: str):
    path = stage1_root / index_type / f"{dataset}_setting{setting}_pqt_results.json"
    return load_json(path)


def load_stage2_summary(stage2_root: Path, dataset: str):
    path = stage2_root / dataset / f"perquery_{dataset}_summary.json"
    payload = load_json(path)
    if "summary" not in payload:
        raise ValueError(f"Stage 2 summary JSON missing 'summary': {path}")
    return payload["summary"]


def maybe_round(value, digits=2):
    if value == "" or value is None:
        return ""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    return round(float(value), digits)


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_stage2_rows(stage2_root: Path):
    rows = []
    for dataset in DATASETS:
        summary = load_stage2_summary(stage2_root, dataset)
        row = {field: maybe_round(summary.get(field)) for field in STAGE2_FIELDS}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize ANN per-query latency results")
    parser.add_argument("--stage1_root", type=Path, required=True)
    parser.add_argument("--stage2_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage2-output", type=Path, required=True)
    parser.add_argument("--compact-output", type=Path)
    args = parser.parse_args()

    stage2_by_dataset = {
        dataset: load_stage2_summary(args.stage2_root, dataset)
        for dataset in DATASETS
    }

    rows = []
    for dataset in DATASETS:
        stage2 = stage2_by_dataset[dataset]
        for setting in SETTINGS:
            k = K_BY_PAIR[(dataset, setting)]
            for index_type in INDEX_TYPES:
                stage1_payload = load_stage1_results(
                    args.stage1_root, dataset, setting, index_type
                )
                stage1 = stage1_payload["results"][index_type]
                stage1_total_ms_mean = (
                    float(stage1["encode_time_per_query_ms_mean"]) +
                    float(stage1["search_time_per_query_ms_mean"])
                )
                stage2_total_ms_mean = float(stage2["total_ms_mean"])
                row = {
                    "dataset": dataset,
                    "setting": setting,
                    "index_type": index_type,
                    "k": k,
                    "stage1_pool_size": stage1_payload["pool_size"],
                    "stage1_num_queries": stage1_payload["num_queries"],
                    "stage1_text_encode_ms_mean": maybe_round(stage1["encode_time_per_query_ms_mean"]),
                    "stage1_text_encode_ms_std": maybe_round(stage1["encode_time_per_query_ms_std"]),
                    "stage1_search_ms_mean": maybe_round(stage1["search_time_per_query_ms_mean"]),
                    "stage1_search_ms_std": maybe_round(stage1["search_time_per_query_ms_std"]),
                    "stage1_total_ms_mean": maybe_round(stage1_total_ms_mean),
                    "stage1_build_s": maybe_round(stage1["build_time_s"]),
                    "stage2_candidate_file": stage2["candidate_file"],
                    "stage2_candidate_count_mean": maybe_round(stage2["candidate_count_mean"]),
                    "stage2_candidate_count_min": maybe_round(stage2["candidate_count_min"]),
                    "stage2_candidate_count_max": maybe_round(stage2["candidate_count_max"]),
                    "stage2_query_encode_ms_mean": maybe_round(stage2["query_encode_ms_mean"]),
                    "stage2_query_encode_ms_std": maybe_round(stage2["query_encode_ms_std"]),
                    "stage2_video_load_ms_mean": maybe_round(stage2["video_load_ms_mean"]),
                    "stage2_video_load_ms_std": maybe_round(stage2["video_load_ms_std"]),
                    "stage2_frame_pooling_ms_mean": maybe_round(stage2["frame_pooling_ms_mean"]),
                    "stage2_frame_pooling_ms_std": maybe_round(stage2["frame_pooling_ms_std"]),
                    "stage2_similarity_compute_ms_mean": maybe_round(stage2["similarity_compute_ms_mean"]),
                    "stage2_similarity_compute_ms_std": maybe_round(stage2["similarity_compute_ms_std"]),
                    "stage2_total_ms_mean": maybe_round(stage2_total_ms_mean),
                    "stage2_total_ms_std": maybe_round(stage2["total_ms_std"]),
                    "end_to_end_total_ms_mean": maybe_round(stage1_total_ms_mean + stage2_total_ms_mean),
                }
                rows.append(row)

    write_csv(args.output, rows, list(rows[0].keys()))
    stage2_rows = build_stage2_rows(args.stage2_root)
    write_csv(args.stage2_output, stage2_rows, STAGE2_FIELDS)

    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Wrote {len(stage2_rows)} rows to {args.stage2_output}")

    if args.compact_output is not None:
        compact_rows = [{field: row[field] for field in COMPACT_FIELDS} for row in rows]
        write_csv(args.compact_output, compact_rows, COMPACT_FIELDS)
        print(f"Wrote {len(compact_rows)} rows to {args.compact_output}")


if __name__ == "__main__":
    main()
