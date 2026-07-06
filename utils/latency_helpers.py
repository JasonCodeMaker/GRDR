"""Shared latency helpers for GRDR and reranker evaluation.

The release scripts keep top-level ``scripts/`` limited to train/eval entry
points. Latency consumers import this module from ``utils/`` instead of the old
figure-generation script tree.
"""
from __future__ import annotations

import json
import socket
import subprocess
import time
from typing import Any

try:
    import torch

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


def _cuda_sync():
    if _HAS_TORCH and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_subset_manifest(path: str) -> dict:
    """Read a latency subset manifest."""
    with open(path) as f:
        return json.load(f)


def host_fingerprint() -> dict:
    """Return a compact one-shot host and GPU fingerprint."""
    info: dict = {"hostname": socket.gethostname()}
    if _HAS_TORCH:
        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            timeout=5,
        ).decode().strip()
        info["nvidia_smi"] = out
    except Exception:
        info["nvidia_smi"] = ""
    return info


class WallCappedTimer:
    """CUDA-synced per-query timer with a wall-time cap."""

    def __init__(
        self,
        warmup_ids,
        timed_ids,
        wall_cap_s: float = 300.0,
        warmup_n_used: int | None = None,
    ):
        self.warmup_ids = list(warmup_ids)
        self.timed_ids = list(timed_ids)
        self.wall_cap_s = float(wall_cap_s)
        if warmup_n_used is None:
            warmup_n_used = len(self.warmup_ids)
        self.warmup_n_used = max(0, min(warmup_n_used, len(self.warmup_ids)))
        self.per_query_s: list[float] = []
        self._t_window_start: float | None = None
        self._cap_hit = False

    def warmup_iter(self):
        for qid in self.warmup_ids[: self.warmup_n_used]:
            yield qid

    def timed_iter(self):
        self._t_window_start = time.perf_counter()
        for qid in self.timed_ids:
            if (time.perf_counter() - self._t_window_start) >= self.wall_cap_s:
                self._cap_hit = True
                break
            yield qid

    class _Measure:
        def __init__(self, parent):
            self.parent = parent
            self.t0 = 0.0

        def __enter__(self):
            _cuda_sync()
            self.t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            _cuda_sync()
            self.parent.per_query_s.append(time.perf_counter() - self.t0)
            return False

    def measure(self):
        return WallCappedTimer._Measure(self)

    def summarize(self) -> dict:
        n_processed = len(self.per_query_s)
        n_target = len(self.timed_ids)
        wall_seconds = (
            time.perf_counter() - self._t_window_start
            if self._t_window_start is not None
            else 0.0
        )
        if n_processed == 0:
            return {
                "n_processed": 0,
                "n_target": n_target,
                "mean_ms": 0.0,
                "p95_ms": 0.0,
                "std_ms": 0.0,
                "wall_seconds": wall_seconds,
                "validity": "failed",
                "cap_hit": self._cap_hit,
            }
        import numpy as _np

        arr_ms = _np.asarray(self.per_query_s, dtype=_np.float64) * 1000.0
        return {
            "n_processed": int(n_processed),
            "n_target": int(n_target),
            "mean_ms": float(arr_ms.mean()),
            "p95_ms": float(_np.percentile(arr_ms, 95)),
            "std_ms": float(arr_ms.std()),
            "wall_seconds": float(wall_seconds),
            "validity": "full_subset" if n_processed >= n_target else "truncated_subset",
            "cap_hit": bool(self._cap_hit),
            "per_query_ms": [float(x) for x in arr_ms],
        }


def build_latency_meta(
    stage_key: str,
    summary: dict,
    manifest: dict,
    warmup_n_used: int,
    extra: dict[str, Any] | None = None,
) -> dict:
    """Build the canonical latency metadata block."""
    block = {
        "online_total_mean": summary["mean_ms"],
        "online_total_p95": summary["p95_ms"],
        "online_total_std": summary["std_ms"],
        "n_processed": summary["n_processed"],
        "n_target": summary["n_target"],
        "warmup_n": len(manifest.get("warmup_query_ids", [])),
        "warmup_n_used": int(warmup_n_used),
        "validity": summary["validity"],
        "wall_seconds": summary["wall_seconds"],
        "cap_hit": summary.get("cap_hit", False),
        "strict_latency_contract": (
            "batch1_candidate_handoff"
            if stage_key == "stage1_latency_ms"
            else "xpool_total_per_query_cuda_sync"
        ),
        "latency_batch_size": 1,
        "subset_manifest_sha256": manifest.get("metadata", {}).get("content_sha256", ""),
        "host_fingerprint": host_fingerprint(),
    }
    if extra:
        block.update(extra)
    return {stage_key: block}
