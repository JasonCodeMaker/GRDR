#!/usr/bin/env python3
"""Helpers for stable Panda-70M downloads on Bunya via Slurm arrays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from download_panda70m_10m import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_COOKIE_FILE,
    DEFAULT_DRIVE_URL,
    DEFAULT_ROOT,
    build_runtime_ydl_opts,
    detect_chrome_user_agent,
    detect_js_runtimes,
    download_metadata,
    extract_metadata,
    find_primary_csv,
    resolve_csv,
    resolve_video2dataset,
    shell_join,
)


def split_csv_round_robin(
    input_csv: Path,
    out_dir: Path,
    *,
    num_splits: int,
    skip_rows: int,
    limit_rows: int | None,
    output_prefix: str | None,
) -> dict[str, object]:
    if num_splits < 1:
        raise ValueError("`num_splits` must be >= 1.")
    if skip_rows < 0:
        raise ValueError("`skip_rows` must be >= 0.")

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_prefix or input_csv.stem
    part_paths = [out_dir / f"{prefix}.part_{idx:03d}.csv" for idx in range(num_splits)]

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"CSV is empty: {input_csv}") from exc

        part_counts = [0 for _ in range(num_splits)]
        writers = []
        handles = []
        try:
            for part_path in part_paths:
                part_handle = part_path.open("w", newline="", encoding="utf-8")
                handles.append(part_handle)
                writer = csv.writer(part_handle, lineterminator="\n")
                writer.writerow(header)
                writers.append(writer)

            total_rows = 0
            for row_idx, row in enumerate(reader):
                if row_idx < skip_rows:
                    continue
                if limit_rows is not None and total_rows >= limit_rows:
                    break
                target = (row_idx - skip_rows) % num_splits
                writers[target].writerow(row)
                part_counts[target] += 1
                total_rows += 1
        finally:
            for part_handle in handles:
                part_handle.close()

    manifest = {
        "input_csv": str(input_csv),
        "out_dir": str(out_dir),
        "num_splits": num_splits,
        "skip_rows": skip_rows,
        "limit_rows": limit_rows,
        "output_prefix": prefix,
        "total_rows_written": sum(part_counts),
        "parts": [
            {
                "task_id": idx,
                "path": str(part_path),
                "rows": part_counts[idx],
            }
            for idx, part_path in enumerate(part_paths)
        ],
    }
    manifest_path = out_dir / f"{prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def cmd_split_csv(
    *,
    input_csv: Path,
    out_dir: Path,
    num_splits: int,
    skip_rows: int,
    limit_rows: int | None,
    output_prefix: str | None,
) -> int:
    manifest = split_csv_round_robin(
        input_csv,
        out_dir,
        num_splits=num_splits,
        skip_rows=skip_rows,
        limit_rows=limit_rows,
        output_prefix=output_prefix,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_prepare_splits(
    *,
    root: Path,
    drive_url: str,
    csv_name: str | None,
    out_dir: Path,
    num_splits: int,
    skip_rows: int,
    limit_rows: int | None,
    output_prefix: str | None,
) -> int:
    zip_path = download_metadata(root, drive_url)
    csv_files = extract_metadata(root, zip_path)
    if csv_name is None:
        input_csv = find_primary_csv(csv_files)
    else:
        input_csv = resolve_csv(root, csv_name)
    manifest = split_csv_round_robin(
        input_csv,
        out_dir,
        num_splits=num_splits,
        skip_rows=skip_rows,
        limit_rows=limit_rows,
        output_prefix=output_prefix,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def cmd_write_runtime_config(
    *,
    base_config_path: Path,
    output_path: Path,
    cookie_file: Path | None,
    user_agent: str | None,
    proxy: str | None,
    processes_count: int,
    thread_count: int,
    sleep_requests: float | None,
    sleep_interval: float | None,
    max_sleep_interval: float | None,
    keep_yt_metadata: bool,
) -> int:
    config = json.loads(json.dumps(yaml.safe_load(base_config_path.read_text())))
    if not isinstance(config, dict):
        raise RuntimeError(f"Unexpected config format in {base_config_path}")

    distribution = config.setdefault("distribution", {})
    distribution["processes_count"] = processes_count
    distribution["thread_count"] = thread_count

    reading = config.setdefault("reading", {})
    yt_args = reading.setdefault("yt_args", {})
    ydl_opts = dict(yt_args.get("ydl_opts") or {})

    resolved_cookie = cookie_file.resolve() if cookie_file is not None else None
    if resolved_cookie is not None and not resolved_cookie.exists():
        raise RuntimeError(f"Cookie file not found: {resolved_cookie}")

    resolved_user_agent = user_agent
    if resolved_user_agent is None and resolved_cookie is not None:
        resolved_user_agent = detect_chrome_user_agent()

    ydl_opts.update(
        build_runtime_ydl_opts(
            cookie_file=resolved_cookie,
            user_agent=resolved_user_agent,
            proxy=proxy,
            sleep_requests=sleep_requests,
            sleep_interval=sleep_interval,
            max_sleep_interval=max_sleep_interval,
            js_runtimes=detect_js_runtimes(),
        )
    )
    if ydl_opts:
        yt_args["ydl_opts"] = ydl_opts
    if not keep_yt_metadata:
        yt_args["yt_metadata_args"] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "cookie_file": str(resolved_cookie) if resolved_cookie is not None else None,
                "user_agent": resolved_user_agent,
                "processes_count": processes_count,
                "thread_count": thread_count,
                "sleep_requests": sleep_requests,
                "sleep_interval": sleep_interval,
                "max_sleep_interval": max_sleep_interval,
                "yt_metadata_enabled": bool(yt_args.get("yt_metadata_args")),
                "js_runtimes": ydl_opts.get("js_runtimes"),
            },
            indent=2,
        )
    )
    return 0


def cmd_print_download_cmd(
    *,
    root: Path,
    csv_path: Path,
    config_path: Path,
    output_dir: Path,
) -> int:
    cmd = [
        str(resolve_video2dataset(root)),
        f"--url_list={csv_path}",
        "--input_format=csv",
        "--url_col=url",
        "--caption_col=caption",
        "--clip_col=timestamp",
        f"--output_folder={output_dir}",
        "--output_format=files",
        "--save_additional_columns=[matching_score,desirable_filtering,shot_boundary_detection]",
        f"--config={config_path}",
    ]
    print(shell_join([str(part) for part in cmd]))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bunya helpers for Panda-70M parallel downloads.")
    parser.add_argument(
        "command",
        choices=[
            "split-csv",
            "prepare-splits",
            "write-runtime-config",
            "print-download-cmd",
        ],
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--drive-url", default=DEFAULT_DRIVE_URL)
    parser.add_argument("--csv-name")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--num-splits", type=int, default=1)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--limit-rows", type=int)
    parser.add_argument("--output-prefix")
    parser.add_argument("--base-config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument(
        "--cookie-file",
        type=Path,
        default=DEFAULT_COOKIE_FILE if DEFAULT_COOKIE_FILE.exists() else None,
    )
    parser.add_argument("--user-agent")
    parser.add_argument("--proxy")
    parser.add_argument("--processes-count", type=int, default=1)
    parser.add_argument("--thread-count", type=int, default=2)
    parser.add_argument("--sleep-requests", type=float)
    parser.add_argument("--sleep-interval", type=float)
    parser.add_argument("--max-sleep-interval", type=float)
    parser.add_argument("--keep-yt-metadata", action="store_true")
    parser.add_argument("--config-path", type=Path)
    parser.add_argument("--csv-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "split-csv":
        if args.input_csv is None or args.out_dir is None:
            raise RuntimeError("`split-csv` requires `--input-csv` and `--out-dir`.")
        return cmd_split_csv(
            input_csv=args.input_csv,
            out_dir=args.out_dir,
            num_splits=args.num_splits,
            skip_rows=args.skip_rows,
            limit_rows=args.limit_rows,
            output_prefix=args.output_prefix,
        )

    if args.command == "prepare-splits":
        if args.out_dir is None:
            raise RuntimeError("`prepare-splits` requires `--out-dir`.")
        return cmd_prepare_splits(
            root=args.root,
            drive_url=args.drive_url,
            csv_name=args.csv_name,
            out_dir=args.out_dir,
            num_splits=args.num_splits,
            skip_rows=args.skip_rows,
            limit_rows=args.limit_rows,
            output_prefix=args.output_prefix,
        )

    if args.command == "write-runtime-config":
        if args.output_path is None:
            raise RuntimeError("`write-runtime-config` requires `--output-path`.")
        return cmd_write_runtime_config(
            base_config_path=args.base_config_path,
            output_path=args.output_path,
            cookie_file=args.cookie_file,
            user_agent=args.user_agent,
            proxy=args.proxy,
            processes_count=args.processes_count,
            thread_count=args.thread_count,
            sleep_requests=args.sleep_requests,
            sleep_interval=args.sleep_interval,
            max_sleep_interval=args.max_sleep_interval,
            keep_yt_metadata=args.keep_yt_metadata,
        )

    if args.command == "print-download-cmd":
        if args.csv_path is None or args.config_path is None or args.output_dir is None:
            raise RuntimeError(
                "`print-download-cmd` requires `--csv-path`, `--config-path`, and `--output-dir`."
            )
        return cmd_print_download_cmd(
            root=args.root,
            csv_path=args.csv_path,
            config_path=args.config_path,
            output_dir=args.output_dir,
        )

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
