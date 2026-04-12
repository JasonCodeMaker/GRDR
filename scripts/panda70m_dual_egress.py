#!/usr/bin/env python3
"""Plan a non-overlapping Panda-70M tail window for the Bunya second egress."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import yaml

from download_panda70m_10m import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_FRAMES_OUTPUT_NAME,
    DEFAULT_RAW_OUTPUT_NAME,
    DEFAULT_ROOT,
    shell_join,
)


DEFAULT_BUNYA_DATA_ROOT = Path("/scratch/project/openps/uqzzha35/Panda-70M-10M")
DEFAULT_BUNYA_REPO_ROOT = Path("/scratch/user/uqzzha35/Project/SemanticID/GRDR")
DEFAULT_BUNYA_ENV_PREFIX = Path("/scratch/project/openps/uqzzha35/conda/envs/panda70m-v2d")


def parse_iso8601(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def resolve_rows_per_shard(config_path: Path) -> int:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise RuntimeError(f"Unexpected config format in {config_path}")
    storage = config.get("storage") or {}
    rows_per_shard = storage.get("number_sample_per_shard")
    if not isinstance(rows_per_shard, int) or rows_per_shard < 1:
        raise RuntimeError(f"`storage.number_sample_per_shard` missing or invalid in {config_path}")
    return rows_per_shard


def processed_dir(root: Path, frames_output_name: str) -> Path:
    return root / "frames" / frames_output_name / "_processed"


def download_dir(root: Path, raw_output_name: str) -> Path:
    return root / "downloads" / raw_output_name


def sorted_numeric_marker_paths(path: Path) -> list[Path]:
    marker_paths = [marker_path for marker_path in path.glob("*.json") if marker_path.stem.isdigit()]
    marker_paths.sort(key=lambda marker_path: int(marker_path.stem))
    return marker_paths


def load_recent_processed_markers(path: Path, recent_count: int) -> tuple[int, list[tuple[int, datetime]]]:
    marker_paths = sorted_numeric_marker_paths(path)
    if not marker_paths:
        raise RuntimeError("No processed shard markers found in the local dataset root.")

    processed_max_shard = int(marker_paths[-1].stem)
    markers: list[tuple[int, datetime]] = []
    for marker_path in marker_paths[-recent_count:]:
        payload = json.loads(marker_path.read_text(encoding="utf-8"))
        processed_at = parse_iso8601(payload.get("processed_at"))
        if processed_at is None:
            continue
        markers.append((int(marker_path.stem), processed_at))
    return processed_max_shard, markers


def max_seen_download_shard(path: Path) -> int | None:
    shard_ids: set[int] = set()
    for pattern in ("*.parquet", "*_stats.json"):
        for candidate in path.glob(pattern):
            shard_id_text = candidate.name.split("_", 1)[0].split(".", 1)[0]
            if len(shard_id_text) != 5 or not shard_id_text.isdigit():
                continue
            shard_ids.add(int(shard_id_text))
    if not shard_ids:
        return None
    return max(shard_ids)


def estimate_shards_per_hour(markers: list[tuple[int, datetime]], recent_count: int) -> float | None:
    recent = markers[-recent_count:]
    if len(recent) < 2:
        return None
    start_shard = recent[0][0]
    end_shard = recent[-1][0]
    start_time = min(ts for _, ts in recent)
    end_time = max(ts for _, ts in recent)
    elapsed_hours = (end_time - start_time).total_seconds() / 3600.0
    if elapsed_hours <= 0:
        return None
    return (end_shard - start_shard) / elapsed_hours


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def cmd_plan_bunya_window(
    *,
    local_root: Path,
    config_path: Path,
    frames_output_name: str,
    raw_output_name: str,
    buffer_hours: float,
    minimum_buffer_shards: int,
    round_up_shards: int,
    recent_count: int,
    window_shards: int,
    array_tasks: int,
    walltime: str,
    bunya_root: Path,
    bunya_repo_root: Path,
    bunya_env_prefix: Path,
    cookie_file: Path,
    processes_count: int,
    thread_count: int,
    sleep_requests: float,
    sleep_interval: float,
    max_sleep_interval: float,
) -> int:
    rows_per_shard = resolve_rows_per_shard(config_path)
    processed_max_shard, markers = load_recent_processed_markers(
        processed_dir(local_root, frames_output_name),
        recent_count,
    )
    download_max_shard = max_seen_download_shard(download_dir(local_root, raw_output_name))
    observed_shards_per_hour = estimate_shards_per_hour(markers, recent_count)
    projected_buffer_shards = (
        int(math.ceil(observed_shards_per_hour * buffer_hours))
        if observed_shards_per_hour is not None
        else 0
    )
    buffer_shards = max(minimum_buffer_shards, projected_buffer_shards)

    next_local_shard = processed_max_shard + 1
    if download_max_shard is not None:
        next_local_shard = max(next_local_shard, download_max_shard + 1)

    recommended_start_shard = round_up(next_local_shard + buffer_shards, round_up_shards)
    recommended_skip_rows = recommended_start_shard * rows_per_shard
    limit_rows = window_shards * rows_per_shard
    run_name = f"dual_egress_s{recommended_start_shard:05d}_n{window_shards:05d}"
    split_prefix = f"panda70m_training_10m.s{recommended_start_shard:05d}"
    split_dir = bunya_root / "metadata" / "splits" / run_name
    logs_dir = bunya_root / "logs" / "slurm" / run_name

    prepare_cmd = [
        str(bunya_env_prefix / "bin" / "python"),
        str(bunya_repo_root / "scripts" / "bunya_panda70m_parallel.py"),
        "prepare-splits",
        "--root",
        str(bunya_root),
        "--csv-name",
        "panda70m_training_10m.csv",
        "--out-dir",
        str(split_dir),
        "--num-splits",
        str(array_tasks),
        "--skip-rows",
        str(recommended_skip_rows),
        "--limit-rows",
        str(limit_rows),
        "--output-prefix",
        split_prefix,
    ]
    submit_cmd = [
        "sbatch",
        "--parsable",
        "--partition=general",
        "--qos=normal",
        f"--time={walltime}",
        f"--array=0-{array_tasks - 1}",
        f"--output={logs_dir}/slurm-%A_%a.out",
        f"--error={logs_dir}/slurm-%A_%a.err",
        "--export",
        ",".join(
            [
                f"ROOT={bunya_root}",
                f"REPO_ROOT={bunya_repo_root}",
                f"ENV_PREFIX={bunya_env_prefix}",
                f"SPLIT_DIR={split_dir}",
                f"SPLIT_PREFIX={split_prefix}",
                f"RUN_NAME={run_name}",
                f"COOKIE_FILE={cookie_file}",
                f"PROCESSES_COUNT={processes_count}",
                f"THREAD_COUNT={thread_count}",
                f"SLEEP_REQUESTS={sleep_requests}",
                f"SLEEP_INTERVAL={sleep_interval}",
                f"MAX_SLEEP_INTERVAL={max_sleep_interval}",
            ]
        ),
        str(bunya_repo_root / "scripts" / "bunya_panda70m_array.sbatch"),
    ]

    payload = {
        "local_root": str(local_root),
        "config_path": str(config_path),
        "rows_per_shard": rows_per_shard,
        "processed_max_shard": processed_max_shard,
        "download_max_shard": download_max_shard,
        "next_local_shard": next_local_shard,
        "observed_shards_per_hour": round(observed_shards_per_hour, 3)
        if observed_shards_per_hour is not None
        else None,
        "buffer_hours": buffer_hours,
        "minimum_buffer_shards": minimum_buffer_shards,
        "projected_buffer_shards": projected_buffer_shards,
        "effective_buffer_shards": buffer_shards,
        "recommended_start_shard": recommended_start_shard,
        "recommended_skip_rows": recommended_skip_rows,
        "window_shards": window_shards,
        "limit_rows": limit_rows,
        "array_tasks": array_tasks,
        "run_name": run_name,
        "split_dir": str(split_dir),
        "split_prefix": split_prefix,
        "prepare_splits_cmd": shell_join(prepare_cmd),
        "submit_cmd": shell_join(submit_cmd),
    }
    print(json.dumps(payload, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan the Bunya second egress for Panda-70M.")
    parser.add_argument("command", choices=["plan-bunya-window"])
    parser.add_argument("--local-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--frames-output-name", default=DEFAULT_FRAMES_OUTPUT_NAME)
    parser.add_argument("--raw-output-name", default=DEFAULT_RAW_OUTPUT_NAME)
    parser.add_argument("--buffer-hours", type=float, default=120.0)
    parser.add_argument("--minimum-buffer-shards", type=int, default=500)
    parser.add_argument("--round-up-shards", type=int, default=100)
    parser.add_argument("--recent-count", type=int, default=200)
    parser.add_argument("--window-shards", type=int, default=1600)
    parser.add_argument("--array-tasks", type=int, default=8)
    parser.add_argument("--walltime", default="5-00:00:00")
    parser.add_argument("--bunya-root", type=Path, default=DEFAULT_BUNYA_DATA_ROOT)
    parser.add_argument("--bunya-repo-root", type=Path, default=DEFAULT_BUNYA_REPO_ROOT)
    parser.add_argument("--bunya-env-prefix", type=Path, default=DEFAULT_BUNYA_ENV_PREFIX)
    parser.add_argument("--cookie-file", type=Path, default=DEFAULT_BUNYA_DATA_ROOT / "cookie.txt")
    parser.add_argument("--processes-count", type=int, default=1)
    parser.add_argument("--thread-count", type=int, default=2)
    parser.add_argument("--sleep-requests", type=float, default=1.0)
    parser.add_argument("--sleep-interval", type=float, default=1.0)
    parser.add_argument("--max-sleep-interval", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "plan-bunya-window":
        return cmd_plan_bunya_window(
            local_root=args.local_root,
            config_path=args.config_path,
            frames_output_name=args.frames_output_name,
            raw_output_name=args.raw_output_name,
            buffer_hours=args.buffer_hours,
            minimum_buffer_shards=args.minimum_buffer_shards,
            round_up_shards=args.round_up_shards,
            recent_count=args.recent_count,
            window_shards=args.window_shards,
            array_tasks=args.array_tasks,
            walltime=args.walltime,
            bunya_root=args.bunya_root,
            bunya_repo_root=args.bunya_repo_root,
            bunya_env_prefix=args.bunya_env_prefix,
            cookie_file=args.cookie_file,
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            sleep_requests=args.sleep_requests,
            sleep_interval=args.sleep_interval,
            max_sleep_interval=args.max_sleep_interval,
        )
    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
