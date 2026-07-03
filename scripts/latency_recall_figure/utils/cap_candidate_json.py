#!/usr/bin/env python
"""Apply a compact candidate handoff cap to a stage-1 candidate JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def candidate_video_id(candidate: Any) -> Any:
    """Return the video id from supported candidate encodings."""
    if isinstance(candidate, dict):
        for key in ("video_id", "video", "vid", "id"):
            if key in candidate:
                return candidate[key]
        return None
    if isinstance(candidate, (list, tuple)):
        if len(candidate) >= 2:
            return candidate[1]
        if len(candidate) == 1:
            return candidate[0]
        return None
    return candidate


def dedup_preserve_order(candidates: list[Any], scores: Any) -> tuple[list[Any], list[Any] | None, list[Any]]:
    """Deduplicate candidates by video id, keeping the first entry and aligned score."""
    has_scores = isinstance(scores, list)
    seen: set[str] = set()
    kept_candidates: list[Any] = []
    kept_scores: list[Any] = []
    kept_video_ids: list[Any] = []
    for idx, candidate in enumerate(candidates):
        video_id = candidate_video_id(candidate)
        if video_id is None:
            continue
        key = str(video_id)
        if key in seen:
            continue
        seen.add(key)
        kept_candidates.append(candidate)
        kept_video_ids.append(video_id)
        if has_scores:
            kept_scores.append(scores[idx] if idx < len(scores) else None)
    return kept_candidates, kept_scores if has_scores else None, kept_video_ids


def percentile(values: list[int], q: float) -> float | None:
    """Nearest-rank percentile for small metric summaries."""
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def hit_rate(results: list[dict[str, Any]], k: int | None = None) -> float | None:
    if not results:
        return None
    hits = 0
    for row in results:
        gt = row.get("ground_truth_video_id")
        video_ids = [candidate_video_id(c) for c in row.get("candidates", [])]
        window = video_ids if k is None else video_ids[:k]
        if gt in window:
            hits += 1
    return 100.0 * hits / len(results)


def apply_cap(payload: dict[str, Any], cap: int, method: str, cap_policy: str) -> dict[str, Any]:
    meta = payload.setdefault("metadata", {})
    if str(meta.get("status", "")).upper() == "OOM":
        return payload

    results = payload.get("results") or []
    pre_counts: list[int] = []
    post_counts: list[int] = []

    for row in results:
        candidates = row.get("candidates") or []
        scores = row.get("scores")
        dedup_candidates, dedup_scores, dedup_video_ids = dedup_preserve_order(candidates, scores)
        pre_counts.append(len(dedup_candidates))
        row.setdefault("pre_cap_num_candidates", len(dedup_candidates))
        row.setdefault("pre_cap_gt_hit", row.get("ground_truth_video_id") in dedup_video_ids)

        capped_candidates = dedup_candidates[:cap]
        row["candidates"] = capped_candidates
        if dedup_scores is not None:
            row["scores"] = dedup_scores[:cap]
        row["num_candidates"] = len(capped_candidates)
        post_counts.append(len(capped_candidates))

    metrics = payload.setdefault("metrics", {})
    metrics.update({
        "total_queries": len(results),
        "candidate_handoff_cap": cap,
        "FullSetHit@All": hit_rate(results, None),
        "CanHit@20": hit_rate(results, 20),
        "CanHit@50": hit_rate(results, 50),
        "CanHit@100": hit_rate(results, 100),
        f"PoolHit@{cap}": hit_rate(results, cap),
        "avg_candidates_per_query": mean(post_counts) if post_counts else None,
        "p95_candidates_per_query": percentile(post_counts, 0.95),
        "max_candidates_per_query": max(post_counts) if post_counts else None,
        "pre_cap_avg_candidates_per_query": mean(pre_counts) if pre_counts else None,
        "pre_cap_p95_candidates_per_query": percentile(pre_counts, 0.95),
        "pre_cap_max_candidates_per_query": max(pre_counts) if pre_counts else None,
    })

    meta.update({
        "method": meta.get("method") or method,
        "candidate_handoff_cap": cap,
        "cap_applied": True,
        "cap_policy": cap_policy,
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--cap", required=True, type=int)
    parser.add_argument("--method", required=True)
    parser.add_argument("--cap-policy", default="compact_candidate_handoff_cap")
    args = parser.parse_args()

    if args.cap <= 0:
        raise SystemExit("--cap must be positive")

    payload = json.loads(args.path.read_text())
    payload = apply_cap(payload, args.cap, args.method, args.cap_policy)
    args.path.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
