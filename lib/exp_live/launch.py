#!/usr/bin/env python3
"""Launch a long-running research command inside the exp-live envelope."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lib.exp_live import harvest  # noqa: E402


@dataclass(frozen=True)
class LaunchResult:
    run_id: str
    run_dir: Path
    meta_path: Path
    live_index: Path


def _utc_stamp(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.UTC).strftime("%Y%m%d-%H%M%S")


def _unique_run_id(exp_id: str, outputs_root: Path, pkg: str, ts: float) -> str:
    base = f"{exp_id}-{_utc_stamp(ts)}"
    run_id = base
    idx = 2
    while (outputs_root / pkg / "runs" / run_id).exists():
        run_id = f"{base}-{idx}"
        idx += 1
    return run_id


def _env_digest(env: dict[str, str]) -> dict[str, str]:
    selected = {
        key: env.get(key, "")
        for key in ("CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "PYTHONPATH")
        if env.get(key)
    }
    payload = json.dumps(selected, sort_keys=True)
    return {"sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "keys": selected}


def _gpu_ids(env: dict[str, str]) -> list[str]:
    value = env.get("CUDA_VISIBLE_DEVICES", "")
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _tmux_session_exists(session: str) -> bool:
    """True when a tmux session with exactly this name is already running."""
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", f"={session}"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"meta.json already exists: {path}")
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tmux_command(
    *,
    run_dir: Path,
    meta_path: Path,
    live_index: Path,
    command: list[str],
    heartbeat_timeout: int,
    total_steps: int | None,
    metrics_regexes: list[str],
    gpu_sample: bool = False,
) -> str:
    harvest_script = Path(__file__).with_name("harvest.py")
    parts = [
        shlex.quote(sys.executable),
        shlex.quote(str(harvest_script)),
        "--run-dir",
        shlex.quote(str(run_dir)),
        "--meta",
        shlex.quote(str(meta_path)),
        "--index",
        shlex.quote(str(live_index)),
        "--heartbeat-timeout",
        str(heartbeat_timeout),
    ]
    if total_steps is not None:
        parts.extend(["--total-steps", str(total_steps)])
    for pattern in metrics_regexes:
        parts.extend(["--metrics-regex", shlex.quote(pattern)])
    if gpu_sample:
        parts.append("--gpu-sample")
    parts.append("--")
    parts.extend(shlex.quote(part) for part in command)
    return " ".join(parts)


def launch_run(
    *,
    pkg: str,
    exp_id: str,
    command: list[str],
    outputs_root: Path = Path("outputs"),
    tmux_session: str | None = None,
    heartbeat_timeout: int = 600,
    total_steps: int | None = None,
    metrics_regexes: list[str] | None = None,
    retry_of: str | None = None,
    telemetry: dict | None = None,
    expected_duration: str | None = None,
    log_adapter: str = "auto",
    gpu_sample: bool = False,
    use_tmux: bool = True,
    now: Callable[[], float] = time.time,
) -> LaunchResult:
    if not command:
        raise ValueError("command is required")
    outputs_root = Path(outputs_root)
    started_at = now()
    run_id = _unique_run_id(exp_id, outputs_root, pkg, started_at)
    run_dir = outputs_root / pkg / "runs" / run_id
    live_index = outputs_root / "_live" / "runs.jsonl"
    meta_path = run_dir / "meta.json"
    session = tmux_session or f"{pkg}-{run_id}".replace("/", "-")[:80]
    # Pre-check before any artifact exists: a doomed launch must leave nothing behind.
    if use_tmux and _tmux_session_exists(session):
        raise RuntimeError(
            f"tmux session already exists: {session!r} — pick another --tmux-session or kill it first"
        )
    env = dict(os.environ)
    meta = {
        "run_id": run_id,
        "pkg": pkg,
        "exp_id": exp_id,
        "command": command,
        "cwd": str(Path.cwd()),
        "env_digest": _env_digest(env),
        "gpu_ids": _gpu_ids(env),
        "tmux_session": session,
        "pid": None,
        "log_path": str(run_dir / "log.txt"),
        "started_at": started_at,
        "retry_of": retry_of,
        "telemetry": telemetry or {},
        "expected_duration_class": expected_duration,
        "log_adapter": log_adapter,
        "transport": "local-tmux",
        "heartbeat_timeout": heartbeat_timeout,
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_meta(meta_path, meta)
    harvest.append_jsonl(live_index, {
        "op": "launched",
        "run_id": run_id,
        "pkg": pkg,
        "exp_id": exp_id,
        "dir": str(run_dir),
        "started_at": started_at,
    })

    metrics_regexes = list(metrics_regexes or [])
    try:
        if use_tmux:
            state = harvest.RunState(run_dir=run_dir, meta=meta, heartbeat_timeout=heartbeat_timeout, total_steps=total_steps)
            state.write_status(now=started_at)
            cmd = _tmux_command(
                run_dir=run_dir,
                meta_path=meta_path,
                live_index=live_index,
                command=command,
                heartbeat_timeout=heartbeat_timeout,
                total_steps=total_steps,
                metrics_regexes=metrics_regexes,
                gpu_sample=gpu_sample,
            )
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", session, "-c", str(Path.cwd()), cmd],
                check=True,
            )
        else:
            harvest.run_command(
                run_dir=run_dir,
                meta=meta,
                command=command,
                heartbeat_timeout=heartbeat_timeout,
                total_steps=total_steps,
                metrics_regexes=metrics_regexes,
                live_index=live_index,
                now=now,
                gpu_sample=gpu_sample,
            )
    except Exception as exc:
        # Safety net: the launched index line is already on disk; pair it with a
        # terminal line so a failed launch can never poison the open-runs stop gate.
        ended_at = now()
        fail_state = harvest.RunState(run_dir=run_dir, meta=meta, heartbeat_timeout=heartbeat_timeout, total_steps=total_steps)
        fail_state.record_fatal(f"launch failed: {exc}")
        fail_state.finalize(exit_code=None, now=ended_at)
        fail_state.write_status(now=ended_at)
        harvest.append_jsonl(live_index, {
            "op": "terminal",
            "run_id": run_id,
            "final_status": "RUN_FAILED",
            "exit_code": None,
            "ended_at": ended_at,
        })
        raise
    return LaunchResult(run_id=run_id, run_dir=run_dir, meta_path=meta_path, live_index=live_index)


def _command_after_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pkg", required=True)
    parser.add_argument("--exp", required=True, dest="exp_id")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--tmux-session")
    parser.add_argument("--log-adapter", default="auto")
    parser.add_argument("--metrics-regex", action="append", default=[])
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--heartbeat-timeout", type=int, default=600)
    parser.add_argument("--retry-of")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--tensorboard-logdir")
    parser.add_argument("--expected-duration", choices=["minutes", "hours", "days"],
                        help="coarse duration class for cadence scheduling only — never recorded as est_time")
    parser.add_argument("--gpu-sample", action="store_true")
    parser.add_argument("--foreground", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = _command_after_separator(args.command)
    if not command:
        parser.error("command is required after --")
    telemetry = {}
    if args.wandb_run_id:
        telemetry["wandb_run_id"] = args.wandb_run_id
    if args.tensorboard_logdir:
        telemetry["tensorboard_logdir"] = args.tensorboard_logdir
    result = launch_run(
        pkg=args.pkg,
        exp_id=args.exp_id,
        command=command,
        outputs_root=Path(args.outputs_root),
        tmux_session=args.tmux_session,
        heartbeat_timeout=args.heartbeat_timeout,
        total_steps=args.total_steps,
        metrics_regexes=args.metrics_regex,
        retry_of=args.retry_of,
        telemetry=telemetry,
        expected_duration=args.expected_duration,
        log_adapter=args.log_adapter,
        gpu_sample=args.gpu_sample,
        use_tmux=not args.foreground,
    )
    print(f"run_id={result.run_id}")
    print(f"run_dir={result.run_dir}")
    print(f"live_index={result.live_index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
