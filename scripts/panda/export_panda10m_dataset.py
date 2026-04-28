#!/usr/bin/env python3
"""Export currently available Panda-70M-10M clips into a readable frame dataset.

Output layout:
  dataset/Panda-70M-10M/
    panda_10m_frames/
      train/<split_prefixed_clip_id>/frame_000.jpg
      val/<split_prefixed_clip_id>/frame_000.jpg
      test/<split_prefixed_clip_id>/frame_000.jpg
    video_retreival_caption/
      panda_10m_ret_train.json
      panda_10m_ret_val.json
      panda_10m_ret_test.json

The export is incremental:
  - Only shards with both a source parquet and a processed tar are considered.
  - Existing frame files are left in place.
  - Metadata JSON is regenerated from the currently usable exported clips.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "pandas with parquet support is required. "
        "Run this script with the Panda-70M-10M download environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = Path("/data2/uqzzha35/VideoRetrieval/Panda-70M-10M")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "dataset" / "Panda-70M-10M"


@dataclass(frozen=True)
class SplitConfig:
    raw_dir: str
    shard_dir: str
    metadata_name: str


SPLIT_CONFIGS: dict[str, SplitConfig] = {
    "train": SplitConfig(
        raw_dir="downloads/train_10m_noaudio_raw",
        shard_dir="frames/train_10m_4f_s256_q4/shards",
        metadata_name="panda_10m_ret_train.json",
    ),
    "val": SplitConfig(
        raw_dir="downloads/val_noaudio_raw",
        shard_dir="frames/val_4f_s256_q4/shards",
        metadata_name="panda_10m_ret_val.json",
    ),
    "test": SplitConfig(
        raw_dir="downloads/test_noaudio_raw",
        shard_dir="frames/test_4f_s256_q4/shards",
        metadata_name="panda_10m_ret_test.json",
    ),
}


def numeric_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    return (int(stem), stem) if stem.isdigit() else (10**18, stem)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def discover_shards(raw_dir: Path, shard_dir: Path) -> list[str]:
    parquet_ids = {path.stem for path in raw_dir.glob("*.parquet")}
    tar_ids = {path.stem for path in shard_dir.glob("*.tar")}
    shard_ids = sorted(parquet_ids & tar_ids, key=lambda value: (int(value), value) if value.isdigit() else (10**18, value))
    return shard_ids


def normalize_scalar(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def load_success_rows(parquet_path: Path) -> list[dict[str, object]]:
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)
    if parquet_path.stat().st_size == 0:
        raise RuntimeError(f"Parquet file is empty: {parquet_path}")

    frame = pd.read_parquet(parquet_path)
    if "status" not in frame.columns:
        raise RuntimeError(f"Missing `status` column in {parquet_path}")
    frame = frame[frame["status"] == "success"]
    if frame.empty:
        return []

    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        normalized = {key: normalize_scalar(value) for key, value in row.items()}
        key = str(normalized["key"])
        caption = normalized.get("caption")
        if caption is None:
            caption = ""
        records.append(
            {
                "key": key,
                "caption": str(caption).strip(),
                "url": normalized.get("url"),
                "clips": normalized.get("clips"),
                "matching_score": normalized.get("matching_score"),
                "desirable_filtering": normalized.get("desirable_filtering"),
                "shot_boundary_detection": normalized.get("shot_boundary_detection"),
            }
        )
    return records


def export_id(split: str, source_key: str) -> str:
    return f"{split}_{source_key}"


def expected_frame_dir(frame_root: Path, split: str, source_key: str) -> Path:
    return frame_root / split / export_id(split, source_key)


def copy_tar_members(
    tar_path: Path,
    *,
    split: str,
    frame_root: Path,
    source_keys: set[str] | None,
) -> tuple[int, int]:
    extracted_files = 0
    clip_dirs = 0
    seen_keys: set[str] = set()

    with tarfile.open(tar_path, "r") as archive:
        for member in archive:
            if not member.isfile():
                continue

            try:
                source_key, relative_name = member.name.split("/", 1)
            except ValueError:
                continue

            if source_keys is not None and source_key not in source_keys:
                continue

            clip_dir = expected_frame_dir(frame_root, split, source_key)
            if source_key not in seen_keys:
                ensure_dir(clip_dir)
                seen_keys.add(source_key)
                clip_dirs += 1

            destination = clip_dir / Path(relative_name).name
            if destination.exists() and destination.stat().st_size > 0:
                continue

            ensure_dir(destination.parent)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to extract {member.name} from {tar_path}")
            with extracted, destination.open("wb") as handle:
                shutil.copyfileobj(extracted, handle)
            extracted_files += 1

    return extracted_files, clip_dirs


def usable_clip_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    frame_files = [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
    return len(frame_files) > 0


def build_metadata_entry(split: str, row: dict[str, object]) -> dict[str, object]:
    clip_name = export_id(split, str(row["key"]))
    return {
        "video": f"{split}/{clip_name}.mp4",
        "caption": row["caption"],
    }


def write_json(path: Path, payload: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def process_direct_shards(
    split: str,
    *,
    shard_root: Path,
    output_root: Path,
    log_every: int,
) -> dict[str, object]:
    frames_root = output_root / "panda_10m_frames"
    ensure_dir(frames_root / split)

    tar_paths = sorted(shard_root.glob("*.tar"), key=numeric_sort_key)
    tar_count = len(tar_paths)
    extracted_files = 0
    extracted_clip_dirs = 0

    for index, tar_path in enumerate(tar_paths, start=1):
        shard_files, shard_clips = copy_tar_members(
            tar_path,
            split=split,
            frame_root=frames_root,
            source_keys=None,
        )
        extracted_files += shard_files
        extracted_clip_dirs += shard_clips

        if log_every > 0 and (index == 1 or index % log_every == 0 or index == tar_count):
            print(
                f"[direct:{split}] shard {index}/{tar_count} "
                f"extracted_files={extracted_files}",
                flush=True,
            )

    return {
        "mode": "direct_shards",
        "split": split,
        "direct_shard_root": str(shard_root),
        "available_shards": tar_count,
        "extracted_clip_dirs_seen": extracted_clip_dirs,
        "extracted_files_written": extracted_files,
        "frames_dir": str(frames_root / split),
    }


def process_split(
    split: str,
    *,
    source_root: Path,
    output_root: Path,
    limit_shards: int | None,
    log_every: int,
) -> dict[str, object]:
    config = SPLIT_CONFIGS[split]
    raw_dir = source_root / config.raw_dir
    shard_dir = source_root / config.shard_dir
    frames_root = output_root / "panda_10m_frames"
    metadata_root = output_root / "video_retreival_caption"

    ensure_dir(frames_root / split)
    ensure_dir(metadata_root)

    shard_ids = discover_shards(raw_dir, shard_dir)
    if limit_shards is not None:
        shard_ids = shard_ids[:limit_shards]

    metadata_entries: list[dict[str, object]] = []
    shard_count = len(shard_ids)
    extracted_files = 0
    extracted_clip_dirs = 0
    skipped_rows = 0
    skipped_shards: list[dict[str, str]] = []

    for index, shard_id in enumerate(shard_ids, start=1):
        parquet_path = raw_dir / f"{shard_id}.parquet"
        tar_path = shard_dir / f"{shard_id}.tar"
        try:
            rows = load_success_rows(parquet_path)
        except Exception as exc:  # pylint: disable=broad-except
            skipped_shards.append({"shard_id": shard_id, "reason": str(exc)})
            print(f"[{split}] skipping shard {shard_id}: {exc}", flush=True)
            continue
        if not rows:
            continue

        source_keys = {str(row["key"]) for row in rows}
        shard_files, shard_clips = copy_tar_members(
            tar_path,
            split=split,
            frame_root=frames_root,
            source_keys=source_keys,
        )
        extracted_files += shard_files
        extracted_clip_dirs += shard_clips

        for row in rows:
            clip_dir = expected_frame_dir(frames_root, split, str(row["key"]))
            if not usable_clip_dir(clip_dir):
                skipped_rows += 1
                continue
            metadata_entries.append(build_metadata_entry(split, row))

        if log_every > 0 and (index == 1 or index % log_every == 0 or index == shard_count):
            print(
                f"[{split}] shard {index}/{shard_count} "
                f"metadata={len(metadata_entries)} extracted_files={extracted_files}",
                flush=True,
            )

    metadata_entries.sort(key=lambda item: item["video"])
    metadata_path = metadata_root / config.metadata_name
    write_json(metadata_path, metadata_entries)

    return {
        "split": split,
        "available_shards": shard_count,
        "metadata_entries": len(metadata_entries),
        "extracted_clip_dirs_seen": extracted_clip_dirs,
        "extracted_files_written": extracted_files,
        "skipped_rows_without_frames": skipped_rows,
        "skipped_shards": skipped_shards,
        "skipped_shard_count": len(skipped_shards),
        "frames_dir": str(frames_root / split),
        "metadata_path": str(metadata_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Panda-70M-10M frames + retrieval metadata.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root directory of the Panda-70M-10M download workspace.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination dataset root under this repository.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=sorted(SPLIT_CONFIGS),
        help="Dataset splits to export.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=None,
        help="Optional shard limit per split for smoke testing.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50,
        help="Progress logging interval in shards.",
    )
    parser.add_argument(
        "--direct-shard-root",
        type=Path,
        default=None,
        help="Directly export every tar under this shard directory without parquet metadata.",
    )
    parser.add_argument(
        "--direct-split",
        choices=sorted(SPLIT_CONFIGS),
        default=None,
        help="Split name to use with --direct-shard-root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.direct_shard_root is not None and args.direct_split is None:
        raise SystemExit("--direct-split is required when --direct-shard-root is set.")
    if args.direct_shard_root is None and args.direct_split is not None:
        raise SystemExit("--direct-split requires --direct-shard-root.")

    summaries = []
    if args.direct_shard_root is not None:
        summaries.append(
            process_direct_shards(
                args.direct_split,
                shard_root=args.direct_shard_root,
                output_root=args.output_root,
                log_every=args.log_every,
            )
        )
    else:
        for split in args.splits:
            summaries.append(
                process_split(
                    split,
                    source_root=args.source_root,
                    output_root=args.output_root,
                    limit_shards=args.limit_shards,
                    log_every=args.log_every,
                )
            )

    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "splits": summaries,
    }
    summary_path = args.output_root / "conversion_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
