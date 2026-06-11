#!/usr/bin/env python3
"""Read bounded exp-live summaries and list open runs."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path
from typing import Callable, Any


TERMINAL = {"COMPLETED", "RUN_FAILED", "RUN_HALTED", "SKIPPED"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tail_lines(path: Path, tail: int) -> list[str]:
    if not path.exists() or tail <= 0:
        return []
    q: deque[str] = deque(maxlen=tail)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            q.append(line.rstrip("\n"))
    return list(q)


def fold_index(outputs_root: Path = Path("outputs")) -> dict[str, dict[str, Any]]:
    folded: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(outputs_root / "_live" / "runs.jsonl"):
        run_id = rec.get("run_id")
        if not run_id:
            continue
        current = folded.setdefault(str(run_id), {})
        if rec.get("op") == "launched":
            current.update(rec)
            current["terminal"] = False
        elif rec.get("op") == "terminal":
            current.update(rec)
            current["terminal"] = True
    return folded


def open_runs(outputs_root: Path = Path("outputs"), now: Callable[[], float] = time.time) -> list[dict[str, Any]]:
    open_items = []
    ts = now()
    for run_id, rec in sorted(fold_index(outputs_root).items()):
        run_dir = Path(rec.get("dir", ""))
        status = _read_json(run_dir / "status.json") if run_dir else {}
        state = status.get("status") or rec.get("final_status") or "RUNNING"
        if rec.get("terminal") or state in TERMINAL:
            continue
        ref = status.get("last_output_at") or status.get("started_at") or rec.get("started_at")
        age = None if ref is None else int(ts - float(ref))
        # A dead harvester leaves status.json frozen at RUNNING; derive STALE from age.
        heartbeat = status.get("heartbeat_timeout") or 600
        if state in {"RUNNING", "QUEUED"} and age is not None and age > heartbeat:
            state = "STALE"
        open_items.append({
            "run_id": run_id,
            "pkg": rec.get("pkg") or status.get("pkg"),
            "exp_id": rec.get("exp_id") or status.get("exp_id"),
            "status": state,
            "last_output_age_s": age,
            "dir": str(run_dir),
        })
    return open_items


def run_summary(run_dir: Path, tail: int = 50) -> dict[str, Any]:
    events = _read_jsonl(run_dir / "events.jsonl")
    anomalies = [event for event in events if event.get("kind") == "anomaly"]
    return {
        "run_dir": str(run_dir),
        "status": _read_json(run_dir / "status.json"),
        "anomalies": anomalies[:5] + anomalies[-5:] if len(anomalies) > 10 else anomalies,
        "tail": _tail_lines(run_dir / "log.txt", tail),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument("--run")
    parser.add_argument("--tail", type=int, default=50)
    parser.add_argument("--open", action="store_true", dest="show_open")
    args = parser.parse_args(argv)

    if args.show_open:
        print(json.dumps(open_runs(Path(args.outputs_root)), indent=2, sort_keys=True))
        return 0
    if args.run:
        print(json.dumps(run_summary(Path(args.run), tail=args.tail), indent=2, sort_keys=True))
        return 0
    parser.error("choose --open or --run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
