#!/usr/bin/env python3
"""Harvest command output into typed experiment-live runtime artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Pattern


RUNNING_STATUSES = {"QUEUED", "RUNNING", "STALE"}
TERMINAL_STATUSES = {"COMPLETED", "RUN_FAILED", "RUN_HALTED", "SKIPPED"}
HARNESS_WRITE_PATHS = (
    "outputs/<pkg>/runs/<run_id>/meta.json",
    "outputs/<pkg>/runs/<run_id>/events.jsonl",
    "outputs/<pkg>/runs/<run_id>/status.json",
    "outputs/<pkg>/runs/<run_id>/log.txt",
    "outputs/_live/runs.jsonl",
)

_NUMBER = r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_KV_RE = re.compile(r"(?P<key>[A-Za-z][A-Za-z0-9_@./-]*)\s*[:=]\s*(?P<value>" + _NUMBER + r")")
_TQDM_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)%\|.*?\|\s*(?P<step>\d+)\s*/\s*(?P<total>\d+)"
    r".*?,\s*(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z/]+)"
)
# Phase lines must look like a marker: a ---/===/### fence, or a P<N>/Epoch/Phase-N lead.
_PHASE_FENCED_RE = re.compile(r"^\s*[-=#]{3,}\s*(?P<label>[^-=#\s].*?)\s*(?:[-=#]{3,}\s*)?$")
_PHASE_MARKER_RE = re.compile(r"^\s*(?P<label>(?:P\d+[a-z]?\b|Epoch\s+\d+\b|Phase\s*\d+\b).*?)\s*$", re.I)
# Word boundaries are load-bearing: bare "Inf"/"NaN" substrings match INFO/inference/etc.
_ANOMALY_RE = re.compile(r"Traceback|(?i:\bCUDA out of memory\b)|(?i:\bout of memory\b)|\bKilled\b|(?i:\bnan\b)|(?i:\binf\b)|(?i:\binfinity\b)")
_FATAL_RE = re.compile(r"Traceback|(?i:\bCUDA out of memory\b)|(?i:\bout of memory\b)|\bKilled\b")


def compile_custom_regex(pattern: str) -> Pattern[str]:
    return re.compile(pattern)


def _json_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def _as_number(value: str) -> int | float:
    parsed = float(value)
    if parsed.is_integer() and not any(ch in value.lower() for ch in (".", "e")):
        return int(parsed)
    return parsed


def _metric_event(values: dict[str, int | float], *, source: str, step: int | None = None, total: int | None = None) -> dict[str, Any]:
    event: dict[str, Any] = {"kind": "metric", "source": source, "step": step, "values": values}
    if total is not None:
        event["total"] = total
    return event


def _parse_jsonl(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    step = _json_number(obj.get("step"))
    total = _json_number(obj.get("total"))
    rate = _json_number(obj.get("rate"))
    epoch = _json_number(obj.get("epoch"))
    unit = obj.get("unit")

    values: dict[str, int | float] = {}
    for key, value in obj.items():
        if key in {"step", "total", "rate", "unit", "epoch", "phase", "kind", "t"}:
            continue
        num = _json_number(value)
        if num is not None:
            values[str(key)] = num

    if step is not None and total is not None and rate is not None and not values:
        event: dict[str, Any] = {
            "kind": "progress",
            "source": "jsonl",
            "step": int(step),
            "total": int(total),
            "rate": float(rate),
            "unit": str(unit or "it/s"),
        }
        if epoch is not None:
            event["epoch"] = int(epoch)
        return event
    if values:
        return _metric_event(values, source="jsonl", step=int(step) if step is not None else None, total=int(total) if total is not None else None)
    return None


def _parse_custom(line: str, regexes: Iterable[Pattern[str]]) -> dict[str, Any] | None:
    for regex in regexes:
        match = regex.search(line)
        if not match:
            continue
        values: dict[str, int | float] = {}
        step = None
        total = None
        for key, value in match.groupdict().items():
            if value is None:
                continue
            try:
                parsed = _as_number(value)
            except ValueError:
                continue
            if key == "step":
                step = int(parsed)
            elif key == "total":
                total = int(parsed)
            else:
                values[key] = parsed
        if values:
            return _metric_event(values, source="custom", step=step, total=total)
    return None


def _parse_tqdm(line: str) -> dict[str, Any] | None:
    match = _TQDM_RE.search(line)
    if not match:
        return None
    return {
        "kind": "progress",
        "source": "tqdm",
        "step": int(match.group("step")),
        "total": int(match.group("total")),
        "rate": float(match.group("rate")),
        "unit": match.group("unit"),
    }


def _parse_kv(line: str) -> dict[str, Any] | None:
    values: dict[str, int | float] = {}
    step = None
    total = None
    rate = None
    for match in _KV_RE.finditer(line):
        key = match.group("key")
        value = _as_number(match.group("value"))
        normalized = key.lower()
        if normalized in {"step", "iter", "iteration"}:
            step = int(value)
        elif normalized == "total":
            total = int(value)
        elif normalized in {"rate", "it/s"}:
            rate = float(value)
        else:
            values[key] = value
    if step is not None and total is not None and rate is not None and not values:
        return {"kind": "progress", "source": "kv-metrics", "step": step, "total": total, "rate": rate, "unit": "it/s"}
    if values:
        return _metric_event(values, source="kv-metrics", step=step, total=total)
    return None


def _parse_phase(line: str) -> dict[str, Any] | None:
    match = _PHASE_FENCED_RE.match(line) or _PHASE_MARKER_RE.match(line)
    if not match:
        return None
    label = match.group("label").strip(" -=#\t")
    if not label:
        return None
    return {"kind": "phase", "label": label}


def _parse_anomaly(line: str) -> dict[str, Any] | None:
    match = _ANOMALY_RE.search(line)
    if not match:
        return None
    label = match.group(0)
    return {
        "kind": "anomaly",
        "label": label,
        "tail": line.rstrip("\n")[-500:],
        "fatal": bool(_FATAL_RE.search(line)),
    }


def parse_line(line: str, custom_regexes: Iterable[Pattern[str]] | None = None) -> dict[str, Any] | None:
    """Parse one output line into at most one typed event."""
    regexes = list(custom_regexes or [])
    for parser in (
        _parse_jsonl,
        lambda text: _parse_custom(text, regexes),
        _parse_tqdm,
        _parse_kv,
        _parse_phase,
        _parse_anomaly,
    ):
        event = parser(line)
        if event:
            return event
    return None


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def exit_status(exit_code: int | None) -> str:
    if exit_code == 0:
        return "COMPLETED"
    if exit_code is not None and exit_code < 0:
        return "RUN_HALTED"
    return "RUN_FAILED"


def gpu_sampler(gpu_ids: list[str], runner: Callable = subprocess.run) -> Callable[[], dict[str, Any] | None]:
    """Build a callable that samples utilization/memory for the run's GPUs via nvidia-smi."""
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]
    if gpu_ids:
        cmd += ["-i", ",".join(gpu_ids)]

    def sample() -> dict[str, Any] | None:
        result = runner(cmd, capture_output=True, text=True, timeout=5)
        lines = (result.stdout or "").strip().splitlines()
        if not lines:
            return None
        util, mem = [part.strip() for part in lines[0].split(",")[:2]]
        return {"gpu_util": float(util), "gpu_mem_gb": round(float(mem) / 1024, 1)}

    return sample


def _format_eta(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {sec}s" if sec else f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m" if minutes else f"{hours}h"


@dataclass
class RunState:
    run_dir: Path
    meta: dict[str, Any]
    heartbeat_timeout: int = 600
    total_steps: int | None = None
    status: str = "RUNNING"
    first_output_at: float | None = None
    last_output_at: float | None = None
    log_lines: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    latest_metrics: dict[str, int | float] = field(default_factory=dict)
    source_map: dict[str, str] = field(default_factory=dict)
    throughput: dict[str, Any] | None = None
    anomalies: int = 0
    resource: dict[str, Any] | None = None
    child_pid: int | None = None
    _fatal_reasons: list[str] = field(default_factory=list)
    _warn_reasons: list[str] = field(default_factory=list)
    _last_warn_at: float | None = None
    exit_code: int | None = None
    ended_at: float | None = None

    def observe_line(self, line: str, now: float) -> None:
        self.log_lines += 1
        if self.first_output_at is None:
            self.first_output_at = now
        self.last_output_at = now

    def apply_event(self, event: dict[str, Any], now: float) -> None:
        if self.first_output_at is None:
            self.first_output_at = now
        self.last_output_at = now
        kind = event.get("kind")
        if kind == "progress":
            total = event.get("total") or self.total_steps
            step = event.get("step")
            rate = event.get("rate")
            unit = event.get("unit") or "it/s"
            if step is not None:
                self.progress["step"] = int(step)
            if total is not None:
                self.progress["total"] = int(total)
            if "epoch" in event:
                self.progress["epoch"] = event["epoch"]
            if total and step is not None:
                self.progress["pct"] = round((float(step) / float(total)) * 100.0, 2)
            if rate is not None:
                if not self.throughput:
                    stable_since = now
                else:
                    stable_since = self.throughput.get("stable_since", now)
                self.throughput = {"rate": float(rate), "unit": str(unit), "stable_since": stable_since}
        elif kind == "metric":
            if event.get("step") is not None:
                self.progress["step"] = int(event["step"])
            total = event.get("total") or self.total_steps
            if total is not None:
                self.progress["total"] = int(total)
                if self.progress.get("step") is not None:
                    self.progress["pct"] = round((float(self.progress["step"]) / float(total)) * 100.0, 2)
            source = event.get("source") or "unknown"
            for key, value in (event.get("values") or {}).items():
                self.latest_metrics[str(key)] = value
                self.source_map[str(key)] = str(source)
                if isinstance(value, float) and not math.isfinite(value):
                    self._record_warn("non-finite metric", now)
        elif kind == "phase":
            self.progress["phase"] = event.get("label")
        elif kind == "anomaly":
            self.anomalies += 1
            label = str(event.get("label") or "anomaly")
            if event.get("fatal"):
                if label not in self._fatal_reasons:
                    self._fatal_reasons.append(label)
            else:
                self._record_warn(label, now)

    def _record_warn(self, reason: str, now: float) -> None:
        if reason not in self._warn_reasons:
            self._warn_reasons.append(reason)
        self._last_warn_at = now

    def record_fatal(self, reason: str) -> None:
        """Record a sticky fatal reason (e.g. a failed launch) outside the event stream."""
        if reason not in self._fatal_reasons:
            self._fatal_reasons.append(reason)

    def finalize(self, exit_code: int | None, now: float) -> None:
        self.exit_code = exit_code
        self.ended_at = now
        self.status = exit_status(exit_code)
        if self.status != "COMPLETED":
            reason = f"exit_code={exit_code}"
            if reason not in self._fatal_reasons:
                self._fatal_reasons.append(reason)

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        ts = time.time() if now is None else now
        status = self.status
        health_state = "OK"
        reasons: list[str] = []

        if status in {"RUNNING", "QUEUED"}:
            ref = self.last_output_at if self.last_output_at is not None else self.meta.get("started_at")
            if ref is not None:
                age = ts - float(ref)
                if age > self.heartbeat_timeout:
                    status = "STALE"
                if age > (self.heartbeat_timeout / 2):
                    health_state = "WARN"
                    reasons.append(f"output silent for {int(age)}s")

        if self._last_warn_at is not None and ts - self._last_warn_at <= (self.heartbeat_timeout / 2):
            health_state = "WARN"
            reasons.extend(self._warn_reasons)

        if self._fatal_reasons or (status in {"RUN_FAILED", "RUN_HALTED"}):
            health_state = "ERROR"
            reasons = self._fatal_reasons or [status]

        eta = "unknown"
        if self.throughput and self.progress.get("total") is not None and self.progress.get("step") is not None:
            stable_for = ts - float(self.throughput.get("stable_since", ts))
            rate = float(self.throughput.get("rate") or 0)
            if stable_for >= 1800 and rate > 0:
                remaining = max(0.0, float(self.progress["total"]) - float(self.progress["step"]))
                eta = _format_eta(remaining / rate)

        return {
            "run_id": self.meta.get("run_id"),
            "pkg": self.meta.get("pkg"),
            "exp_id": self.meta.get("exp_id"),
            "status": status,
            "health": {"state": health_state, "reasons": reasons},
            "progress": dict(self.progress),
            "latest_metrics": dict(self.latest_metrics),
            "source_map": dict(self.source_map),
            "throughput": self.throughput,
            "eta": eta,
            "first_output_at": self.first_output_at,
            "last_output_at": self.last_output_at,
            "started_at": self.meta.get("started_at"),
            "heartbeat_timeout": self.heartbeat_timeout,
            "anomalies": self.anomalies,
            "log_lines": self.log_lines,
            "resource": self.resource,
            "pid": self.child_pid,
            "harvester_pid": os.getpid(),
            "exit_code": self.exit_code,
            "ended_at": self.ended_at,
        }

    def write_status(self, now: float | None = None) -> dict[str, Any]:
        status = self.snapshot(now=now)
        atomic_json(self.run_dir / "status.json", status)
        return status


def run_command(
    *,
    run_dir: Path,
    meta: dict[str, Any],
    command: list[str],
    heartbeat_timeout: int = 600,
    total_steps: int | None = None,
    metrics_regexes: Iterable[str] | None = None,
    live_index: Path | None = None,
    now: Callable[[], float] = time.time,
    status_interval: float = 1.0,
    watchdog_interval: float | None = None,
    gpu_sample: bool = False,
    sampler: Callable[[], dict[str, Any] | None] | None = None,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    regexes = [compile_custom_regex(pattern) for pattern in (metrics_regexes or [])]
    state = RunState(run_dir=run_dir, meta=meta, heartbeat_timeout=heartbeat_timeout, total_steps=total_steps)
    if gpu_sample and sampler is None:
        sampler = gpu_sampler([str(g) for g in (meta.get("gpu_ids") or [])])
    lock = threading.Lock()
    with lock:
        state.write_status(now=now())
    last_write = now()

    # Watchdog: re-write the snapshot on a wall clock so STALE/WARN appear during
    # output silence — the exact case the line-driven loop below cannot cover.
    if watchdog_interval is None:
        watchdog_interval = min(15.0, max(0.5, heartbeat_timeout / 4))
    stop = threading.Event()

    def _watchdog() -> None:
        while not stop.wait(watchdog_interval):
            if sampler is not None:
                try:
                    state.resource = sampler()
                except Exception:
                    state.resource = None
            with lock:
                state.write_status(now=time.time())

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()

    log_path = run_dir / "log.txt"
    events_path = run_dir / "events.jsonl"
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        with lock:
            state.child_pid = proc.pid
            state.write_status(now=now())
        with log_path.open("a", encoding="utf-8") as log:
            assert proc.stdout is not None
            for line in proc.stdout:
                ts = now()
                log.write(line)
                log.flush()
                with lock:
                    state.observe_line(line, now=ts)
                    force = state.log_lines == 1
                event = parse_line(line, regexes)
                if event:
                    event = {"t": ts, **event}
                    append_jsonl(events_path, event)
                    with lock:
                        state.apply_event(event, now=ts)
                    force = force or event.get("kind") == "anomaly"
                # Throttle: chatty output must not mean one rename per line.
                if force or ts - last_write >= status_interval:
                    with lock:
                        state.write_status(now=ts)
                    last_write = ts
        exit_code = proc.wait()
    finally:
        stop.set()
        watchdog.join(timeout=watchdog_interval + 1)
    ended_at = now()
    with lock:
        state.finalize(exit_code, now=ended_at)
        final_status = state.write_status(now=ended_at)
    if live_index is not None:
        append_jsonl(live_index, {
            "op": "terminal",
            "run_id": meta.get("run_id"),
            "final_status": final_status["status"],
            "exit_code": exit_code,
            "ended_at": ended_at,
        })
    return exit_code


def _command_after_separator(command: list[str]) -> list[str]:
    if command and command[0] == "--":
        return command[1:]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--index")
    parser.add_argument("--heartbeat-timeout", type=int, default=600)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--metrics-regex", action="append", default=[])
    parser.add_argument("--gpu-sample", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = _command_after_separator(args.command)
    if not command:
        parser.error("command is required after --")
    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    return run_command(
        run_dir=Path(args.run_dir),
        meta=meta,
        command=command,
        heartbeat_timeout=args.heartbeat_timeout,
        total_steps=args.total_steps,
        metrics_regexes=args.metrics_regex,
        live_index=Path(args.index) if args.index else None,
        gpu_sample=args.gpu_sample,
    )


if __name__ == "__main__":
    raise SystemExit(main())
