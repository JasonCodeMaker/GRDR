#!/usr/bin/env python3
"""Prepare and run the Panda-70M 10M download-to-frames pipeline.

Pipeline summary:
1. Download Panda metadata and extract the official CSV files.
2. Launch `video2dataset` inside tmux to download no-audio clips into shard folders.
3. Watch for completed shards, uniformly extract 4 JPEG frames per clip with ffmpeg,
   package frames into one tar per shard, and delete the raw clip shard on success.

The final on-disk strategy intentionally keeps shard-level parquet/stats files and
stores frames as shard tarballs. JPEGs are already compressed, so we avoid adding a
second compression layer that would cost CPU without much space gain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/data2/uqzzha35/VideoRetrieval/Panda-70M-10M")
DEFAULT_DRIVE_URL = "https://drive.google.com/file/d/1LLOFeYw9nZzjT5aA1Wj4oGi5yHUzwSk5/view?usp=sharing"
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "panda70m" / "panda70m_10m_noaudio_balanced.yaml"
DEFAULT_COOKIE_FILE = REPO_ROOT / "dataset" / "Panda-70M-10M" / "cookie.txt"
DEFAULT_RAW_OUTPUT_NAME = "train_10m_noaudio_raw"
DEFAULT_FRAMES_OUTPUT_NAME = "train_10m_4f_s256_q4"
DEFAULT_SESSION_NAME = "panda70m_10m"
DEFAULT_NUM_FRAMES = 4
DEFAULT_MIN_SIDE = 256
DEFAULT_JPEG_QUALITY = 4
DEFAULT_POLL_SECONDS = 30
DEFAULT_MIN_FREE_GB = 20.0
DEFAULT_FRAME_JOBS = max(1, min(6, (os.cpu_count() or 6) // 2))
DEFAULT_DOWNLOAD_PROCESSES = 6
DEFAULT_DOWNLOAD_THREADS = 4
DEFAULT_SLEEP_REQUESTS = None
DEFAULT_SLEEP_INTERVAL = None
DEFAULT_MAX_SLEEP_INTERVAL = None
DEFAULT_FALLBACK_CHROME_VERSION = "146.0.0.0"


@dataclass(frozen=True)
class FrameOutput:
    clip_key: str
    frames: list[Path]


def run(
    cmd: list[str],
    *,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=check,
        text=True,
        capture_output=capture_output,
    )


def shell_join(parts: list[str]) -> str:
    return shlex.join([str(part) for part in parts])


def ensure_layout(root: Path) -> None:
    paths = [
        root / "metadata",
        root / "downloads",
        root / "frames",
        root / "logs",
        root / "logs" / "tmux",
        root / "tmp",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def resolve_gdown() -> str | None:
    return shutil.which("gdown")


def resolve_video2dataset(root: Path) -> Path:
    local_path = root / ".venv" / "bin" / "video2dataset"
    if local_path.exists():
        return local_path

    global_path = shutil.which("video2dataset")
    if global_path:
        return Path(global_path)

    raise RuntimeError(
        "`video2dataset` not found. Expected either it in PATH or at "
        f"{local_path}."
    )


def status_payload(root: Path, config_path: Path) -> dict[str, object]:
    zip_path = root / "metadata" / "panda70m_train_10m.zip"
    extracted = list((root / "metadata").glob("*.csv"))
    usage = shutil.disk_usage(root)
    try:
        video2dataset_path = str(resolve_video2dataset(root))
    except RuntimeError:
        video2dataset_path = None

    return {
        "root": str(root),
        "config_path": str(config_path),
        "default_cookie_file": str(DEFAULT_COOKIE_FILE) if DEFAULT_COOKIE_FILE.exists() else None,
        "gdown": resolve_gdown(),
        "video2dataset": video2dataset_path,
        "ffmpeg": shutil.which("ffmpeg"),
        "ffprobe": shutil.which("ffprobe"),
        "tmux": shutil.which("tmux"),
        "metadata_zip_exists": zip_path.exists(),
        "metadata_zip_path": str(zip_path),
        "csv_files": [str(p) for p in sorted(extracted)],
        "disk_free_gb": round(usage.free / (1024**3), 2),
    }


def detect_chrome_user_agent() -> str:
    candidates = [
        ["google-chrome", "--version"],
        ["chromium", "--version"],
        ["chromium-browser", "--version"],
        ["microsoft-edge", "--version"],
    ]
    for cmd in candidates:
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        version = result.stdout.strip().split()[-1]
        if version and version[0].isdigit():
            return (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                f"(KHTML, like Gecko) Chrome/{version} Safari/537.36"
            )
    return (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        f"(KHTML, like Gecko) Chrome/{DEFAULT_FALLBACK_CHROME_VERSION} Safari/537.36"
    )


def detect_js_runtimes() -> dict[str, dict[str, str]]:
    candidates = [
        ("deno", "deno"),
        ("node", "node"),
        ("bun", "bun"),
        ("quickjs", "qjs"),
    ]
    for runtime_name, executable in candidates:
        if shutil.which(executable):
            return {runtime_name: {}}
    return {}


def build_runtime_ydl_opts(
    *,
    cookie_file: Path | None,
    user_agent: str | None,
    proxy: str | None,
    sleep_requests: float | None,
    sleep_interval: float | None,
    max_sleep_interval: float | None,
    js_runtimes: dict[str, dict[str, str]],
) -> dict[str, object]:
    ydl_opts: dict[str, object] = {}
    if cookie_file is not None:
        ydl_opts["cookiefile"] = str(cookie_file)
    if user_agent:
        ydl_opts["http_headers"] = {"User-Agent": user_agent}
    if proxy:
        ydl_opts["proxy"] = proxy
    if sleep_requests is not None:
        ydl_opts["sleep_interval_requests"] = sleep_requests
    if sleep_interval is not None:
        ydl_opts["sleep_interval"] = sleep_interval
    if max_sleep_interval is not None:
        ydl_opts["max_sleep_interval"] = max_sleep_interval
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes
    return ydl_opts


def write_runtime_download_config(
    base_config_path: Path,
    runtime_config_path: Path,
    *,
    cookie_file: Path | None,
    user_agent: str | None,
    proxy: str | None,
    processes_count: int,
    thread_count: int,
    sleep_requests: float | None,
    sleep_interval: float | None,
    max_sleep_interval: float | None,
    keep_yt_metadata: bool,
    js_runtimes: dict[str, dict[str, str]],
) -> dict[str, object]:
    config = yaml.safe_load(base_config_path.read_text())
    if not isinstance(config, dict):
        raise RuntimeError(f"Unexpected config format in {base_config_path}")

    distribution = config.setdefault("distribution", {})
    distribution["processes_count"] = processes_count
    distribution["thread_count"] = thread_count

    reading = config.setdefault("reading", {})
    yt_args = reading.setdefault("yt_args", {})
    ydl_opts = dict(yt_args.get("ydl_opts") or {})
    ydl_opts.update(
        build_runtime_ydl_opts(
            cookie_file=cookie_file,
            user_agent=user_agent,
            proxy=proxy,
            sleep_requests=sleep_requests,
            sleep_interval=sleep_interval,
            max_sleep_interval=max_sleep_interval,
            js_runtimes=js_runtimes,
        )
    )
    if ydl_opts:
        yt_args["ydl_opts"] = ydl_opts

    if not keep_yt_metadata:
        yt_args["yt_metadata_args"] = {}

    runtime_config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config


def cmd_check(root: Path, config_path: Path) -> int:
    ensure_layout(root)
    print(json.dumps(status_payload(root, config_path), indent=2))
    return 0


def download_metadata(root: Path, drive_url: str) -> Path:
    ensure_layout(root)
    zip_path = root / "metadata" / "panda70m_train_10m.zip"
    if zip_path.exists():
        return zip_path

    gdown = resolve_gdown()
    if not gdown:
        raise RuntimeError("`gdown` not found in PATH.")

    subprocess.run([gdown, "--fuzzy", drive_url, "-O", str(zip_path)], check=True)
    return zip_path


def extract_metadata(root: Path, zip_path: Path) -> list[Path]:
    metadata_dir = root / "metadata"
    csv_files = sorted(metadata_dir.glob("*.csv"))
    if csv_files:
        return csv_files

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(metadata_dir)

    csv_files = sorted(metadata_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV files found after extracting {zip_path}")
    return csv_files


def find_primary_csv(csv_files: list[Path]) -> Path:
    preferred = [p for p in csv_files if "train" in p.name.lower() and "test" not in p.name.lower()]
    return sorted(preferred or csv_files)[0]


def resolve_csv(root: Path, csv_name: str | None) -> Path:
    metadata_dir = root / "metadata"
    if csv_name:
        candidate = Path(csv_name)
        if not candidate.is_absolute():
            candidate = metadata_dir / csv_name
        if not candidate.exists():
            raise RuntimeError(f"CSV file not found: {candidate}")
        return candidate

    csv_files = sorted(metadata_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("No extracted CSV metadata found. Run `prepare` first.")
    return find_primary_csv(csv_files)


def render_video2dataset_cmd(
    root: Path,
    url_list: Path,
    *,
    config_path: Path,
    raw_output_name: str,
) -> list[str]:
    output_dir = root / "downloads" / raw_output_name
    return [
        str(resolve_video2dataset(root)),
        f"--url_list={url_list}",
        "--input_format=csv",
        "--url_col=url",
        "--caption_col=caption",
        "--clip_col=timestamp",
        f"--output_folder={output_dir}",
        "--output_format=files",
        "--save_additional_columns=[matching_score,desirable_filtering,shot_boundary_detection]",
        f"--config={config_path}",
    ]


def cmd_prepare(root: Path, drive_url: str, config_path: Path) -> int:
    zip_path = download_metadata(root, drive_url)
    csv_files = extract_metadata(root, zip_path)
    primary_csv = find_primary_csv(csv_files)
    payload = status_payload(root, config_path)
    payload["primary_csv"] = str(primary_csv)
    payload["download_command"] = shell_join(
        render_video2dataset_cmd(
            root,
            primary_csv,
            config_path=config_path,
            raw_output_name=DEFAULT_RAW_OUTPUT_NAME,
        )
    )
    print(json.dumps(payload, indent=2))
    return 0


def cmd_print_download(root: Path, config_path: Path, raw_output_name: str, csv_name: str | None) -> int:
    csv_path = resolve_csv(root, csv_name)
    print(
        shell_join(
            render_video2dataset_cmd(
                root,
                csv_path,
                config_path=config_path,
                raw_output_name=raw_output_name,
            )
        )
    )
    return 0


def probe_video(video_path: Path) -> tuple[float, float]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate:format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    result = run(cmd)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    avg_frame_rate = streams[0].get("avg_frame_rate", "0/0")
    try:
        fps = float(Fraction(avg_frame_rate))
    except (ZeroDivisionError, ValueError):
        fps = 0.0

    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    return max(fps, 1.0), max(duration, 0.04)


def compute_uniform_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    total_frames = max(1, total_frames)
    indices = []
    for i in range(num_frames):
        idx = int(round(((i + 0.5) * total_frames / num_frames) - 0.5))
        idx = max(0, min(total_frames - 1, idx))
        indices.append(idx)
    return indices


def scale_filter(min_side: int) -> str:
    return f"scale='if(lt(iw,ih),{min_side},-2)':'if(lt(iw,ih),-2,{min_side})'"


def extract_uniform_frames(
    video_path: Path,
    output_dir: Path,
    *,
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
) -> list[Path]:
    fps, duration = probe_video(video_path)
    estimated_total_frames = max(1, int(round(fps * duration)))
    indices = compute_uniform_frame_indices(estimated_total_frames, num_frames)

    unique_indices: list[int] = []
    for idx in indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
    if not unique_indices:
        unique_indices = [0]

    select_expr = "+".join([f"eq(n\\,{idx})" for idx in unique_indices])
    output_pattern = output_dir / "extract_%03d.jpg"
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select='{select_expr}',{scale_filter(min_side)}",
        "-vsync",
        "0",
        "-q:v",
        str(jpeg_quality),
        str(output_pattern),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    extracted = sorted(output_dir.glob("extract_*.jpg"))
    if not extracted:
        fallback = output_dir / "extract_001.jpg"
        fallback_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            scale_filter(min_side),
            "-frames:v",
            "1",
            "-q:v",
            str(jpeg_quality),
            str(fallback),
        ]
        subprocess.run(fallback_cmd, check=True)
        extracted = [fallback]

    if len(extracted) > num_frames:
        extracted = extracted[:num_frames]

    while len(extracted) < num_frames:
        duplicate = output_dir / f"dup_{len(extracted):03d}.jpg"
        shutil.copy2(extracted[-1], duplicate)
        extracted.append(duplicate)

    return extracted


def extract_clip_frames(
    video_path: Path,
    temp_root: Path,
    *,
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
) -> FrameOutput:
    clip_key = video_path.stem
    clip_dir = temp_root / clip_key
    clip_dir.mkdir(parents=True, exist_ok=False)
    frames = extract_uniform_frames(
        video_path,
        clip_dir,
        num_frames=num_frames,
        min_side=min_side,
        jpeg_quality=jpeg_quality,
    )
    return FrameOutput(clip_key=clip_key, frames=frames)


def frame_dataset_paths(root: Path, frames_output_name: str) -> dict[str, Path]:
    frame_root = root / "frames" / frames_output_name
    paths = {
        "frame_root": frame_root,
        "shards": frame_root / "shards",
        "processed": frame_root / "_processed",
        "failed": frame_root / "_failed",
        "tmp": frame_root / "_tmp",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def completed_shard_dirs(raw_dir: Path) -> list[Path]:
    shard_dirs = []
    for child in sorted(raw_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue
        if (raw_dir / f"{child.name}.parquet").exists() and (raw_dir / f"{child.name}_stats.json").exists():
            shard_dirs.append(child)
    return shard_dirs


def has_active_shards(raw_dir: Path) -> bool:
    for child in raw_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            if not (raw_dir / f"{child.name}.parquet").exists():
                return True
    return False


def find_incomplete_shard_ids(raw_dir: Path) -> list[str]:
    shard_ids: set[str] = set()
    for child in raw_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            shard_id = child.name
            if not (raw_dir / f"{shard_id}_stats.json").exists():
                shard_ids.add(shard_id)
    for child in raw_dir.iterdir():
        if not child.is_file() or child.suffix != ".parquet":
            continue
        shard_id = child.stem
        if shard_id.isdigit() and not (raw_dir / f"{shard_id}_stats.json").exists():
            shard_ids.add(shard_id)
    return sorted(shard_ids)


def quarantine_incomplete_shards(raw_dir: Path) -> dict[str, object]:
    shard_ids = find_incomplete_shard_ids(raw_dir)
    if not shard_ids:
        return {"count": 0, "quarantine_dir": None, "shard_ids": []}

    quarantine_dir = raw_dir / f"_incomplete_quarantine_{time.strftime('%Y%m%d_%H%M%S')}"
    quarantine_dir.mkdir(parents=True, exist_ok=False)

    for shard_id in shard_ids:
        for candidate in [
            raw_dir / shard_id,
            raw_dir / f"{shard_id}.parquet",
            raw_dir / f"{shard_id}_stats.json",
        ]:
            if candidate.exists():
                shutil.move(str(candidate), str(quarantine_dir / candidate.name))

    return {
        "count": len(shard_ids),
        "quarantine_dir": str(quarantine_dir),
        "shard_ids": shard_ids,
    }


def process_completed_shard(
    shard_dir: Path,
    *,
    frame_paths: dict[str, Path],
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
    jobs: int,
) -> dict[str, object]:
    shard_id = shard_dir.name
    processed_marker = frame_paths["processed"] / f"{shard_id}.json"
    failed_marker = frame_paths["failed"] / f"{shard_id}.json"
    frame_tar = frame_paths["shards"] / f"{shard_id}.tar"
    temp_tar = frame_paths["shards"] / f"{shard_id}.tar.tmp"

    if processed_marker.exists():
        return {"shard_id": shard_id, "status": "already_processed"}
    if failed_marker.exists():
        return {"shard_id": shard_id, "status": "failed_marker_present"}

    temp_root = Path(tempfile.mkdtemp(prefix=f"{shard_id}_", dir=frame_paths["tmp"]))
    mp4_files = sorted(shard_dir.glob("*.mp4"))
    clip_outputs: list[FrameOutput] = []
    failures: list[dict[str, str]] = []
    try:
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
            future_map = {
                executor.submit(
                    extract_clip_frames,
                    video_path,
                    temp_root,
                    num_frames=num_frames,
                    min_side=min_side,
                    jpeg_quality=jpeg_quality,
                ): video_path
                for video_path in mp4_files
            }
            for future in as_completed(future_map):
                video_path = future_map[future]
                try:
                    clip_outputs.append(future.result())
                except Exception as exc:  # pylint: disable=broad-except
                    failures.append({"video": video_path.name, "error": str(exc)})

        if failures:
            payload = {
                "shard_id": shard_id,
                "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "failures": failures,
            }
            failed_marker.write_text(json.dumps(payload, indent=2))
            raise RuntimeError(f"{len(failures)} clips failed while processing shard {shard_id}")

        if temp_tar.exists():
            temp_tar.unlink()
        if frame_tar.exists():
            frame_tar.unlink()

        with tarfile.open(temp_tar, "w") as tar:
            for output in sorted(clip_outputs, key=lambda item: item.clip_key):
                for idx, frame_path in enumerate(output.frames):
                    tar.add(
                        frame_path,
                        arcname=f"{output.clip_key}/frame_{idx:03d}.jpg",
                        recursive=False,
                    )
        temp_tar.replace(frame_tar)

        processed_payload = {
            "shard_id": shard_id,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "frame_tar": str(frame_tar),
            "clip_count": len(mp4_files),
            "sample_metadata": str(shard_dir.parent / f"{shard_id}.parquet"),
            "stats_file": str(shard_dir.parent / f"{shard_id}_stats.json"),
        }
        processed_marker.write_text(json.dumps(processed_payload, indent=2))
        shutil.rmtree(shard_dir)
        return processed_payload
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def tmux_session_exists(session_name: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", session_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def choose_tmux_session_name(base_name: str) -> str:
    if not tmux_session_exists(base_name):
        return base_name
    suffix = time.strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{suffix}"


def make_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | 0o111)


def write_tmux_runner(
    path: Path,
    *,
    command: str,
    log_path: Path,
    done_file: Path | None = None,
) -> None:
    worker_path = path.with_name(f"{path.stem}.worker.sh")
    pid_path = path.with_name(f"{path.stem}.pid")
    status_path = path.with_name(f"{path.stem}.status")

    done_lines = ""
    if done_file is not None:
        done_lines = f"touch {shlex.quote(str(done_file))}"

    worker_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -uo pipefail
        trap '' HUP

        set +e
        {command} >> {shlex.quote(str(log_path))} 2>&1
        status=$?
        set -e
        (
            {done_lines}
            echo "[done] $(date -Is) status=${{exit_status}}" | tee -a {shlex.quote(str(log_path))}
        ) 2>/dev/null
        printf '%s\n' "${{status}}" > {shlex.quote(str(status_path))}
        rm -f {shlex.quote(str(pid_path))}
        exit "${{status}}"
        """
    )
    worker_script = worker_script.replace("${exit_status}", "${status}")
    worker_path.write_text(worker_script)
    make_executable(worker_path)

    runner_script = textwrap.dedent(
        f"""\
        #!/usr/bin/env bash
        set -euo pipefail

        attach_existing() {{
            local existing_pid="$1"
            echo "[resume] $(date -Is) pid=${{existing_pid}}" | tee -a {shlex.quote(str(log_path))}
            tail --pid="${{existing_pid}}" -n 20 -F {shlex.quote(str(log_path))} || true
            if [[ -f {shlex.quote(str(status_path))} ]]; then
                exit "$(cat {shlex.quote(str(status_path))})"
            fi
            exit 0
        }}

        mkdir -p {shlex.quote(str(log_path.parent))}
        if [[ -f {shlex.quote(str(pid_path))} ]]; then
            existing_pid="$(cat {shlex.quote(str(pid_path))} 2>/dev/null || true)"
            if [[ -n "${{existing_pid}}" ]] && kill -0 "${{existing_pid}}" 2>/dev/null; then
                attach_existing "${{existing_pid}}"
            fi
        fi

        rm -f {shlex.quote(str(status_path))}
        echo "[start] $(date -Is)" | tee -a {shlex.quote(str(log_path))}
        nohup bash {shlex.quote(str(worker_path))} >/dev/null 2>&1 &
        child_pid=$!
        printf '%s\n' "${{child_pid}}" > {shlex.quote(str(pid_path))}
        tail --pid="${{child_pid}}" -n 0 -F {shlex.quote(str(log_path))} || true
        wait "${{child_pid}}" 2>/dev/null || true
        if [[ -f {shlex.quote(str(status_path))} ]]; then
            exit "$(cat {shlex.quote(str(status_path))})"
        fi
        exit 1
        """
    )
    path.write_text(runner_script)
    make_executable(path)


def read_runtime_concurrency(runtime_config_path: Path) -> tuple[int | None, int | None]:
    if not runtime_config_path.exists():
        return None, None
    try:
        config = json.loads(runtime_config_path.read_text())
    except json.JSONDecodeError:
        return None, None
    if not isinstance(config, dict):
        return None, None
    distribution = config.get("distribution", {})
    if not isinstance(distribution, dict):
        return None, None
    return (
        distribution.get("processes_count"),
        distribution.get("thread_count"),
    )


def kill_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return


def stop_tmux_workers(tmux_dir: Path) -> None:
    for name in ("download.pid", "postprocess.pid"):
        pid_path = tmux_dir / name
        if not pid_path.exists():
            continue
        try:
            pid = int(pid_path.read_text().strip())
        except ValueError:
            continue
        kill_process_group(pid)


def count_new_errors(log_path: Path, offset: int, patterns: list[str]) -> tuple[int, dict[str, int]]:
    if not log_path.exists():
        return offset, {pat: 0 for pat in patterns}
    with log_path.open("r", errors="replace") as handle:
        handle.seek(offset)
        chunk = handle.read()
        new_offset = handle.tell()
    counts = {pat: chunk.count(pat) for pat in patterns}
    return new_offset, counts


def free_space_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def cmd_watch_frames(
    root: Path,
    *,
    raw_output_name: str,
    frames_output_name: str,
    done_file: Path,
    jobs: int,
    poll_seconds: int,
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
    min_free_gb: float,
    download_tmux_target: str | None,
) -> int:
    raw_dir = root / "downloads" / raw_output_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = frame_dataset_paths(root, frames_output_name)

    download_stopped_for_space = False

    while True:
        free_gb = free_space_gb(root)
        if free_gb < min_free_gb:
            print(
                f"[watcher] free space {free_gb:.2f} GB below threshold {min_free_gb:.2f} GB",
                flush=True,
            )
            if download_tmux_target and not download_stopped_for_space:
                subprocess.run(
                    ["tmux", "send-keys", "-t", download_tmux_target, "C-c"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                download_stopped_for_space = True
                print(
                    f"[watcher] sent Ctrl-C to downloader target {download_tmux_target}",
                    flush=True,
                )

        pending = []
        for shard_dir in completed_shard_dirs(raw_dir):
            shard_id = shard_dir.name
            if (paths["processed"] / f"{shard_id}.json").exists():
                continue
            if (paths["failed"] / f"{shard_id}.json").exists():
                continue
            pending.append(shard_dir)

        if pending:
            for shard_dir in pending:
                print(f"[watcher] processing shard {shard_dir.name}", flush=True)
                try:
                    result = process_completed_shard(
                        shard_dir,
                        frame_paths=paths,
                        num_frames=num_frames,
                        min_side=min_side,
                        jpeg_quality=jpeg_quality,
                        jobs=jobs,
                    )
                    print(json.dumps(result, indent=2), flush=True)
                except Exception as exc:  # pylint: disable=broad-except
                    print(
                        f"[watcher] shard {shard_dir.name} failed: {exc}",
                        flush=True,
                    )
        else:
            downloader_done = done_file.exists()
            active_shards = has_active_shards(raw_dir)
            if downloader_done and not active_shards:
                print("[watcher] downloader finished and no active shards remain", flush=True)
                return 0
            time.sleep(max(1, poll_seconds))


def cmd_launch_tmux(
    root: Path,
    *,
    config_path: Path,
    drive_url: str,
    csv_name: str | None,
    raw_output_name: str,
    frames_output_name: str,
    session_name: str,
    jobs: int,
    poll_seconds: int,
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
    min_free_gb: float,
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
    if shutil.which("tmux") is None:
        raise RuntimeError("`tmux` not found in PATH.")

    ensure_layout(root)
    zip_path = download_metadata(root, drive_url)
    extract_metadata(root, zip_path)
    csv_path = resolve_csv(root, csv_name)
    effective_session_name = choose_tmux_session_name(session_name)

    tmux_dir = root / "logs" / "tmux" / effective_session_name
    tmux_dir.mkdir(parents=True, exist_ok=True)
    done_file = tmux_dir / "download.done"
    download_log = tmux_dir / "download.log"
    postprocess_log = tmux_dir / "postprocess.log"
    download_script = tmux_dir / "download.sh"
    postprocess_script = tmux_dir / "postprocess.sh"
    runtime_config_path = tmux_dir / "download_config.runtime.json"

    if done_file.exists():
        done_file.unlink()

    raw_dir = root / "downloads" / raw_output_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    incomplete_quarantine = quarantine_incomplete_shards(raw_dir)

    resolved_cookie_file = cookie_file.resolve() if cookie_file is not None else None
    if resolved_cookie_file is not None and not resolved_cookie_file.exists():
        raise RuntimeError(f"Cookie file not found: {resolved_cookie_file}")

    resolved_user_agent = user_agent
    if resolved_user_agent is None and resolved_cookie_file is not None:
        resolved_user_agent = detect_chrome_user_agent()
    resolved_js_runtimes = detect_js_runtimes()

    runtime_config = write_runtime_download_config(
        config_path,
        runtime_config_path,
        cookie_file=resolved_cookie_file,
        user_agent=resolved_user_agent,
        proxy=proxy,
        processes_count=processes_count,
        thread_count=thread_count,
        sleep_requests=sleep_requests,
        sleep_interval=sleep_interval,
        max_sleep_interval=max_sleep_interval,
        keep_yt_metadata=keep_yt_metadata,
        js_runtimes=resolved_js_runtimes,
    )

    download_cmd = shell_join(
        render_video2dataset_cmd(
            root,
            csv_path,
            config_path=runtime_config_path,
            raw_output_name=raw_output_name,
        )
    )
    write_tmux_runner(
        download_script,
        command=download_cmd,
        log_path=download_log,
        done_file=done_file,
    )

    postprocess_cmd = shell_join(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "watch-frames",
            f"--root={root}",
            f"--raw-output-name={raw_output_name}",
            f"--frames-output-name={frames_output_name}",
            f"--done-file={done_file}",
            f"--jobs={jobs}",
            f"--poll-seconds={poll_seconds}",
            f"--num-frames={num_frames}",
            f"--min-side={min_side}",
            f"--jpeg-quality={jpeg_quality}",
            f"--min-free-gb={min_free_gb}",
            f"--download-tmux-target={effective_session_name}:download",
        ]
    )
    write_tmux_runner(
        postprocess_script,
        command=postprocess_cmd,
        log_path=postprocess_log,
        done_file=None,
    )

    subprocess.run(["tmux", "new-session", "-d", "-s", effective_session_name, "-n", "download"], check=True)
    subprocess.run(["tmux", "set-option", "-t", effective_session_name, "remain-on-exit", "on"], check=True)
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            f"{effective_session_name}:download",
            f"bash {shlex.quote(str(download_script))}",
            "Enter",
        ],
        check=True,
    )
    subprocess.run(["tmux", "new-window", "-t", effective_session_name, "-n", "postprocess"], check=True)
    subprocess.run(
        [
            "tmux",
            "send-keys",
            "-t",
            f"{effective_session_name}:postprocess",
            f"bash {shlex.quote(str(postprocess_script))}",
            "Enter",
        ],
        check=True,
    )

    payload = {
        "session_name": effective_session_name,
        "attach_command": f"tmux attach -t {effective_session_name}",
        "download_window": f"{effective_session_name}:download",
        "postprocess_window": f"{effective_session_name}:postprocess",
        "csv_path": str(csv_path),
        "config_path": str(runtime_config_path),
        "base_config_path": str(config_path),
        "raw_output_dir": str(root / "downloads" / raw_output_name),
        "frames_output_dir": str(root / "frames" / frames_output_name),
        "download_log": str(download_log),
        "postprocess_log": str(postprocess_log),
        "disk_free_gb_before_launch": round(free_space_gb(root), 2),
        "cookie_file": str(resolved_cookie_file) if resolved_cookie_file is not None else None,
        "user_agent": resolved_user_agent,
        "proxy": proxy,
        "js_runtimes": runtime_config["reading"]["yt_args"].get("ydl_opts", {}).get("js_runtimes"),
        "processes_count": runtime_config["distribution"]["processes_count"],
        "thread_count": runtime_config["distribution"]["thread_count"],
        "sleep_requests": runtime_config["reading"]["yt_args"].get("ydl_opts", {}).get("sleep_interval_requests"),
        "sleep_interval": runtime_config["reading"]["yt_args"].get("ydl_opts", {}).get("sleep_interval"),
        "max_sleep_interval": runtime_config["reading"]["yt_args"].get("ydl_opts", {}).get("max_sleep_interval"),
        "yt_metadata_enabled": bool(runtime_config["reading"]["yt_args"].get("yt_metadata_args")),
        "incomplete_quarantine": incomplete_quarantine,
    }
    print(json.dumps(payload, indent=2))
    return 0


def cmd_auto_tune(
    root: Path,
    *,
    config_path: Path,
    drive_url: str,
    csv_name: str | None,
    raw_output_name: str,
    frames_output_name: str,
    session_name: str,
    jobs: int,
    poll_seconds: int,
    num_frames: int,
    min_side: int,
    jpeg_quality: int,
    min_free_gb: float,
    cookie_file: Path | None,
    user_agent: str | None,
    proxy: str | None,
    min_processes: int,
    max_processes: int,
    min_threads: int,
    max_threads: int,
    step_processes: int,
    step_threads: int,
    check_seconds: int,
    error_threshold: int,
    stable_increase_count: int,
    keep_yt_metadata: bool,
) -> int:
    if shutil.which("tmux") is None:
        raise RuntimeError("`tmux` not found in PATH.")

    ensure_layout(root)
    zip_path = download_metadata(root, drive_url)
    extract_metadata(root, zip_path)
    resolve_csv(root, csv_name)

    tmux_dir = root / "logs" / "tmux" / session_name
    tmux_dir.mkdir(parents=True, exist_ok=True)
    download_log = tmux_dir / "download.log"
    runtime_config_path = tmux_dir / "download_config.runtime.json"

    current_proc, current_thread = read_runtime_concurrency(runtime_config_path)
    if current_proc is None:
        current_proc = min_processes
    if current_thread is None:
        current_thread = min_threads

    patterns = [
        "Sign in to confirm",
        "Please sign in",
        "HTTP Error 429",
        "No such file or directory",
        "Traceback",
    ]

    def relaunch(proc_count: int, thread_count: int) -> None:
        stop_tmux_workers(tmux_dir)
        if tmux_session_exists(session_name):
            subprocess.run(["tmux", "kill-session", "-t", session_name], check=False)
        cmd_launch_tmux(
            root,
            config_path=config_path,
            drive_url=drive_url,
            csv_name=csv_name,
            raw_output_name=raw_output_name,
            frames_output_name=frames_output_name,
            session_name=session_name,
            jobs=jobs,
            poll_seconds=poll_seconds,
            num_frames=num_frames,
            min_side=min_side,
            jpeg_quality=jpeg_quality,
            min_free_gb=min_free_gb,
            cookie_file=cookie_file,
            user_agent=user_agent,
            proxy=proxy,
            processes_count=proc_count,
            thread_count=thread_count,
            sleep_requests=None,
            sleep_interval=None,
            max_sleep_interval=None,
            keep_yt_metadata=keep_yt_metadata,
        )

    relaunch(current_proc, current_thread)
    offset = download_log.stat().st_size if download_log.exists() else 0
    stable_windows = 0
    raw_dir = root / "downloads" / raw_output_name
    last_stats = len(list(raw_dir.glob("*_stats.json")))

    while True:
        time.sleep(max(30, check_seconds))
        offset, counts = count_new_errors(download_log, offset, patterns)
        total_errors = sum(counts.values())

        current_stats = len(list(raw_dir.glob("*_stats.json")))
        progressed = current_stats > last_stats
        last_stats = current_stats

        if total_errors >= error_threshold:
            stable_windows = 0
            if current_thread > min_threads:
                current_thread = max(min_threads, current_thread - step_threads)
            elif current_proc > min_processes:
                current_proc = max(min_processes, current_proc - step_processes)
            relaunch(current_proc, current_thread)
            offset = download_log.stat().st_size if download_log.exists() else 0
            continue

        if progressed:
            stable_windows += 1
        else:
            stable_windows = 0

        if stable_windows >= stable_increase_count:
            stable_windows = 0
            if current_proc < max_processes:
                current_proc = min(max_processes, current_proc + step_processes)
            elif current_thread < max_threads:
                current_thread = min(max_threads, current_thread + step_threads)
            relaunch(current_proc, current_thread)
            offset = download_log.stat().st_size if download_log.exists() else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure and run the Panda-70M 10M frame pipeline.")
    parser.add_argument(
        "command",
        choices=[
            "check",
            "prepare",
            "print-download-cmd",
            "watch-frames",
            "launch-tmux",
            "auto-tune",
        ],
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--drive-url", default=DEFAULT_DRIVE_URL)
    parser.add_argument("--config-path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--csv-name")
    parser.add_argument("--raw-output-name", default=DEFAULT_RAW_OUTPUT_NAME)
    parser.add_argument("--frames-output-name", default=DEFAULT_FRAMES_OUTPUT_NAME)
    parser.add_argument("--session-name", default=DEFAULT_SESSION_NAME)
    parser.add_argument("--done-file", type=Path)
    parser.add_argument("--jobs", type=int, default=DEFAULT_FRAME_JOBS)
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES)
    parser.add_argument("--min-side", type=int, default=DEFAULT_MIN_SIDE)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB)
    parser.add_argument("--download-tmux-target")
    parser.add_argument("--cookie-file", type=Path, default=DEFAULT_COOKIE_FILE if DEFAULT_COOKIE_FILE.exists() else None)
    parser.add_argument("--user-agent")
    parser.add_argument("--proxy")
    parser.add_argument("--processes-count", type=int, default=DEFAULT_DOWNLOAD_PROCESSES)
    parser.add_argument("--thread-count", type=int, default=DEFAULT_DOWNLOAD_THREADS)
    parser.add_argument("--sleep-requests", type=float, default=DEFAULT_SLEEP_REQUESTS)
    parser.add_argument("--sleep-interval", type=float, default=DEFAULT_SLEEP_INTERVAL)
    parser.add_argument("--max-sleep-interval", type=float, default=DEFAULT_MAX_SLEEP_INTERVAL)
    parser.add_argument("--keep-yt-metadata", action="store_true")
    parser.add_argument("--min-processes", type=int, default=1)
    parser.add_argument("--max-processes", type=int, default=DEFAULT_DOWNLOAD_PROCESSES)
    parser.add_argument("--min-threads", type=int, default=1)
    parser.add_argument("--max-threads", type=int, default=DEFAULT_DOWNLOAD_THREADS)
    parser.add_argument("--step-processes", type=int, default=1)
    parser.add_argument("--step-threads", type=int, default=1)
    parser.add_argument("--check-seconds", type=int, default=300)
    parser.add_argument("--error-threshold", type=int, default=1)
    parser.add_argument("--stable-increase-count", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "check":
            return cmd_check(args.root, args.config_path)
        if args.command == "prepare":
            return cmd_prepare(args.root, args.drive_url, args.config_path)
        if args.command == "print-download-cmd":
            return cmd_print_download(args.root, args.config_path, args.raw_output_name, args.csv_name)
        if args.command == "watch-frames":
            done_file = args.done_file
            if done_file is None:
                raise RuntimeError("`watch-frames` requires `--done-file`.")
            return cmd_watch_frames(
                args.root,
                raw_output_name=args.raw_output_name,
                frames_output_name=args.frames_output_name,
                done_file=done_file,
                jobs=args.jobs,
                poll_seconds=args.poll_seconds,
                num_frames=args.num_frames,
                min_side=args.min_side,
                jpeg_quality=args.jpeg_quality,
                min_free_gb=args.min_free_gb,
                download_tmux_target=args.download_tmux_target,
            )
        if args.command == "launch-tmux":
            return cmd_launch_tmux(
                args.root,
                config_path=args.config_path,
                drive_url=args.drive_url,
                csv_name=args.csv_name,
                raw_output_name=args.raw_output_name,
                frames_output_name=args.frames_output_name,
                session_name=args.session_name,
                jobs=args.jobs,
                poll_seconds=args.poll_seconds,
                num_frames=args.num_frames,
                min_side=args.min_side,
                jpeg_quality=args.jpeg_quality,
                min_free_gb=args.min_free_gb,
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
        if args.command == "auto-tune":
            return cmd_auto_tune(
                args.root,
                config_path=args.config_path,
                drive_url=args.drive_url,
                csv_name=args.csv_name,
                raw_output_name=args.raw_output_name,
                frames_output_name=args.frames_output_name,
                session_name=args.session_name,
                jobs=args.jobs,
                poll_seconds=args.poll_seconds,
                num_frames=args.num_frames,
                min_side=args.min_side,
                jpeg_quality=args.jpeg_quality,
                min_free_gb=args.min_free_gb,
                cookie_file=args.cookie_file,
                user_agent=args.user_agent,
                proxy=args.proxy,
                min_processes=args.min_processes,
                max_processes=args.max_processes,
                min_threads=args.min_threads,
                max_threads=args.max_threads,
                step_processes=args.step_processes,
                step_threads=args.step_threads,
                check_seconds=args.check_seconds,
                error_threshold=args.error_threshold,
                stable_increase_count=args.stable_increase_count,
                keep_yt_metadata=args.keep_yt_metadata,
            )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
