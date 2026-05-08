#!/usr/bin/env python3
"""Regenerate only high-risk Panda pseudo queries with safer prompts."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from generate_panda_pseudo_queries import (
    PandaFrameDataset,
    collate_panda_batch,
    generate_group,
    load_model_and_processor,
    normalize_caption,
)


GRDR_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption"
DEFAULT_SOURCE_JSON = DATA_ROOT / "panda_10m_ret_train.json"
DEFAULT_BASE_JSON = DATA_ROOT / "panda_10m_ret_train_addition.pre_repair_20260422.json"
DEFAULT_OUTPUT_JSON = DATA_ROOT / "panda_10m_ret_train_addition.regen_model.json"
DEFAULT_PROGRESS_JSONL = DATA_ROOT / "panda_10m_ret_train_addition.regen_model.progress.jsonl"
DEFAULT_REPORT_JSON = DATA_ROOT / "panda_10m_ret_train_addition.regen_model.report.json"
DEFAULT_FAILURES_JSONL = DATA_ROOT / "panda_10m_ret_train_addition.regen_model.failures.jsonl"
DEFAULT_FRAMES_ROOT = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "panda_10m_frames" / "train"

BAD_ENDINGS = {"and", "or", "with", "of", "in", "on", "at", "to", "for", "from", "the", "a", "an", "one", "their", "his", "her", "there"}
SENSITIVE_TOKENS = {
    "abc",
    "boston",
    "bristol",
    "cbs",
    "cnn",
    "daniel",
    "duncker",
    "fox",
    "ghul",
    "governor",
    "jimbo",
    "mills",
    "molloy",
    "mormon",
    "msnbc",
    "palin",
    "pam",
    "sarah",
    "schaefer",
    "stingray",
    "superman",
    "trebuchet",
    "wabc",
}

PROMPT_GROUPS = [
    {
        "prompt": (
            "Write a concise video search query using only directly visible content from this clip. "
            "Mention action, objects, and scene. Do not guess names, titles, channels, brands, locations, "
            "characters, or backstory."
        ),
        "temperature": 1.00,
        "top_p": 0.82,
        "num_return_sequences": 3,
    },
    {
        "prompt": (
            "Write a different retrieval query for this clip focusing on visible people, objects, clothing, "
            "or setting details only. Use different wording. No identities unless visible as on-screen text."
        ),
        "temperature": 1.12,
        "top_p": 0.86,
        "num_return_sequences": 3,
    },
    {
        "prompt": (
            "Write another diverse search query for this clip using only what can be seen. "
            "If readable text is clearly visible, you may mention the visible text exactly; otherwise do not invent text."
        ),
        "temperature": 1.22,
        "top_p": 0.88,
        "num_return_sequences": 3,
    },
    {
        "prompt": (
            "Write a distinct short query for this clip that highlights a unique visible detail or composition. "
            "Avoid proper names, organizations, events, and inferred story context."
        ),
        "temperature": 1.32,
        "top_p": 0.90,
        "num_return_sequences": 3,
    },
]

FALLBACK_PROMPT_GROUPS = [
    {
        "prompt": (
            "Write one more safe retrieval query for this clip using only visible content. "
            "No names, no brands, no channels, no invented text, and no trailing incomplete phrase."
        ),
        "temperature": 1.10,
        "top_p": 0.84,
        "num_return_sequences": 4,
    },
    {
        "prompt": (
            "Write a new concise search query for this clip with fresh wording and visible details only. "
            "Do not infer identity, place, event, or story."
        ),
        "temperature": 1.20,
        "top_p": 0.87,
        "num_return_sequences": 4,
    },
]


def content_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_caption(text))


def is_flagged(query: str, original: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    q_words = content_words(query)
    if q_words and q_words[-1] in BAD_ENDINGS:
        reasons.append("dangling")

    original_norm = normalize_caption(original)
    query_norm = normalize_caption(query)
    for token in SENSITIVE_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", query_norm) and not re.search(rf"\b{re.escape(token)}\b", original_norm):
            reasons.append(f"sensitive:{token}")

    return bool(reasons), reasons


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            done.add(json.loads(line)["video"])
    return done


def append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_fallback_pool(original: str) -> list[str]:
    base = normalize_caption(original)
    stripped = re.sub(r"^(a|an|the)\s+", "", base).strip() or base
    pool = [
        base,
        stripped,
        f"video of {stripped}",
        f"clip of {stripped}",
        f"scene of {stripped}",
        f"showing {stripped}",
        f"featuring {stripped}",
        f"footage of {stripped}",
    ]
    out: list[str] = []
    seen: set[str] = set()
    for item in pool:
        norm = normalize_caption(item)
        if not norm or norm in seen:
            continue
        if content_words(norm) and content_words(norm)[-1] in BAD_ENDINGS:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def generate_candidate_pool(model, frames, videos, original_by_video, stopping_criteria) -> dict[str, list[str]]:
    pool = {video: [] for video in videos}
    seen = {video: set() for video in videos}

    for group in PROMPT_GROUPS + FALLBACK_PROMPT_GROUPS:
        prompts = [group["prompt"]] * len(videos)
        group_outputs = generate_group(
            model,
            frames,
            prompts,
            stopping_criteria=stopping_criteria,
            temperature=group["temperature"],
            top_p=group["top_p"],
            num_return_sequences=group["num_return_sequences"],
            max_new_tokens=40,
        )
        for video, outputs in zip(videos, group_outputs):
            original = original_by_video[video]
            for output in outputs:
                norm = normalize_caption(output)
                if not norm or norm in seen[video]:
                    continue
                flagged, _ = is_flagged(norm, original)
                if flagged:
                    continue
                if len(content_words(norm)) <= 3:
                    continue
                seen[video].add(norm)
                pool[video].append(norm)

    return pool


def fill_video_queries(
    video: str,
    original: str,
    items: list[tuple[int, str]],
    candidate_pool: list[str],
) -> tuple[list[dict[str, object]], int, int]:
    existing_good: set[str] = set()
    output_rows: list[dict[str, object]] = []
    model_count = 0
    fallback_count = 0

    flagged_rows: list[tuple[int, str]] = []
    for idx, caption in items:
        flagged, reasons = is_flagged(caption, original)
        if flagged:
            flagged_rows.append((idx, caption))
            output_rows.append({"index": idx, "video": video, "caption": None, "reasons": reasons})
        else:
            norm = normalize_caption(caption)
            existing_good.add(norm)
            output_rows.append({"index": idx, "video": video, "caption": caption, "reasons": []})

    usable_model = [cand for cand in candidate_pool if cand not in existing_good]
    fallback_pool = [cand for cand in build_fallback_pool(original) if cand not in existing_good]

    candidate_ptr = 0
    fallback_ptr = 0
    for row in output_rows:
        if row["caption"] is not None:
            continue
        replacement = None
        while candidate_ptr < len(usable_model):
            cand = usable_model[candidate_ptr]
            candidate_ptr += 1
            if cand not in existing_good:
                replacement = cand
                model_count += 1
                break
        if replacement is None:
            while fallback_ptr < len(fallback_pool):
                cand = fallback_pool[fallback_ptr]
                fallback_ptr += 1
                if cand not in existing_good:
                    replacement = cand
                    fallback_count += 1
                    break
        if replacement is None:
            base = normalize_caption(original)
            suffix = len(existing_good) + 1
            while True:
                cand = normalize_caption(f"{base} variation {suffix}")
                suffix += 1
                if cand not in existing_good:
                    replacement = cand
                    fallback_count += 1
                    break
        existing_good.add(replacement)
        row["caption"] = replacement

    output_rows.sort(key=lambda row: row["index"])
    return output_rows, model_count, fallback_count


def finalize_output(base_entries, progress_path: Path, output_path: Path) -> int:
    replacement_by_index: dict[int, str] = {}
    with progress_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for item in rec["rows"]:
                replacement_by_index[item["index"]] = item["caption"]

    out = []
    for idx, item in enumerate(base_entries):
        caption = replacement_by_index.get(idx, item["caption"])
        out.append({"video": item["video"], "caption": caption})

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, output_path)
    return len(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--base-json", type=Path, default=DEFAULT_BASE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--progress-jsonl", type=Path, default=DEFAULT_PROGRESS_JSONL)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--failures-jsonl", type=Path, default=DEFAULT_FAILURES_JSONL)
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite:
        for path in (args.output_json, args.progress_jsonl, args.report_json, args.failures_jsonl):
            path.unlink(missing_ok=True)

    with args.source_json.open() as f:
        source_entries = json.load(f)
    with args.base_json.open() as f:
        base_entries = json.load(f)

    original_by_video = {item["video"]: item["caption"] for item in source_entries}
    grouped = defaultdict(list)
    flagged_video_records = []
    reason_counts: Counter[str] = Counter()
    total_flagged_queries = 0

    for idx, item in enumerate(base_entries):
        video = item["video"]
        grouped[video].append((idx, item["caption"]))

    for video, items in grouped.items():
        flagged_indices = []
        for idx, caption in items:
            flagged, reasons = is_flagged(caption, original_by_video[video])
            if flagged:
                flagged_indices.append(idx)
                total_flagged_queries += 1
                for reason in reasons:
                    reason_counts[reason] += 1
        if flagged_indices:
            flagged_video_records.append({"video": video})

    if args.max_videos is not None:
        flagged_video_records = flagged_video_records[: args.max_videos]

    done = load_done(args.progress_jsonl)
    todo = [record for record in flagged_video_records if record["video"] not in done]

    print(
        json.dumps(
            {
                "flagged_videos_total": len(flagged_video_records),
                "flagged_queries_total": total_flagged_queries,
                "done_videos": len(done),
                "todo_videos": len(todo),
                "reason_counts": dict(sorted(reason_counts.items())),
            },
            indent=2,
        )
    )

    if todo:
        model, vis_processor, stopping_criteria = load_model_and_processor()

        dataset = PandaFrameDataset(todo, args.frames_root, vis_processor.transform)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_panda_batch,
            persistent_workers=args.num_workers > 0,
        )

        model_replacements = 0
        fallback_replacements = 0
        processed_videos = len(done)
        pbar = tqdm(loader, desc="regen-flagged")
        for batch in pbar:
            if batch["invalid"]:
                append_jsonl(args.failures_jsonl, batch["invalid"])

            frames = batch["frames"]
            videos = batch["videos"]
            if frames is None or not videos:
                continue

            frames = frames.to("cuda", non_blocking=True)
            candidate_pool = generate_candidate_pool(model, frames, videos, original_by_video, stopping_criteria)

            records_to_append = []
            for video in videos:
                rows, model_count, fallback_count = fill_video_queries(
                    video,
                    original_by_video[video],
                    grouped[video],
                    candidate_pool[video],
                )
                model_replacements += model_count
                fallback_replacements += fallback_count
                records_to_append.append(
                    {
                        "video": video,
                        "rows": rows,
                    }
                )

            append_jsonl(args.progress_jsonl, records_to_append)
            processed_videos += len(videos)
            pbar.set_postfix(done=processed_videos, model=model_replacements, fallback=fallback_replacements)

    output_entries = finalize_output(base_entries, args.progress_jsonl, args.output_json)
    processed_videos = len(load_done(args.progress_jsonl))
    report = {
        "base_json": str(args.base_json),
        "output_json": str(args.output_json),
        "output_entries": output_entries,
        "flagged_videos_total": len(flagged_video_records),
        "flagged_queries_total": total_flagged_queries,
        "processed_videos": processed_videos,
        "reason_counts": dict(sorted(reason_counts.items())),
    }
    tmp_report = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    with tmp_report.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    os.replace(tmp_report, args.report_json)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
