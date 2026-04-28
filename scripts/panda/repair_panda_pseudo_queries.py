#!/usr/bin/env python3
"""Repair high-risk Panda pseudo queries by rewriting from the source caption."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


GRDR_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption"
DEFAULT_SOURCE_JSON = DATA_ROOT / "panda_10m_ret_train.json"
DEFAULT_INPUT_JSON = DATA_ROOT / "panda_10m_ret_train_addition.json"
DEFAULT_OUTPUT_JSON = DATA_ROOT / "panda_10m_ret_train_addition.repaired.json"
DEFAULT_REPORT_JSON = DATA_ROOT / "panda_10m_ret_train_addition.repair_report.json"
DEFAULT_REPLACEMENTS_JSONL = DATA_ROOT / "panda_10m_ret_train_addition.replacements.jsonl"

BAD_ENDINGS = {"and", "or", "with", "of", "in", "on", "at", "to", "for", "from", "the", "a", "an", "one", "their", "his", "her", "there"}
SENSITIVE_TOKENS = {
    "abc",
    "boston",
    "bristol",
    "cbs",
    "cnn",
    "fox",
    "governor",
    "jimbo",
    "mills",
    "molloy",
    "mormon",
    "msnbc",
    "palin",
    "sarah",
    "stingray",
    "superman",
    "trebuchet",
    "wabc",
}
LOW_PRECISION_THRESHOLD = 0.30
MIN_CONTENT_TOKENS = 4
CONTENT_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "around",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "down",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "off",
    "on",
    "or",
    "out",
    "over",
    "group",
    "man",
    "men",
    "people",
    "person",
    "speaker",
    "someone",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "through",
    "to",
    "under",
    "up",
    "video",
    "was",
    "were",
    "while",
    "woman",
    "women",
    "with",
}


def normalize_text(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.?!;:,]+$", "", text)
    return text.lower().strip()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", normalize_text(text))


def content_tokens(text: str) -> list[str]:
    return [word for word in words(text) if word not in CONTENT_STOPWORDS]


def contains_token(text: str, token: str) -> bool:
    return re.search(rf"\b{re.escape(token)}\b", normalize_text(text)) is not None


@dataclass
class FlagResult:
    flagged: bool
    reasons: list[str]


def flag_query(query: str, original: str) -> FlagResult:
    q_words = words(query)
    reasons: list[str] = []

    if q_words and q_words[-1] in BAD_ENDINGS:
        reasons.append("dangling")

    original_norm = normalize_text(original)
    for token in SENSITIVE_TOKENS:
        if contains_token(query, token) and not contains_token(original_norm, token):
            reasons.append(f"sensitive:{token}")

    query_content = set(content_tokens(query))
    original_content = set(content_tokens(original))
    overlap = len(query_content & original_content)
    precision = overlap / (len(query_content) or 1)
    if len(query_content) >= MIN_CONTENT_TOKENS and precision <= LOW_PRECISION_THRESHOLD:
        reasons.append("low_precision")

    return FlagResult(flagged=bool(reasons), reasons=reasons)


def drop_leading_phrase(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):].strip()
    return text


def make_phrase(original: str) -> str:
    text = normalize_text(original)
    for prefix in (
        "there is ",
        "there are ",
        "it is ",
        "the video is about ",
        "this video is about ",
        "the clip is about ",
        "this clip is about ",
    ):
        text = drop_leading_phrase(text, prefix)

    if not text:
        text = normalize_text(original)
    return text


def compress_phrase(phrase: str) -> str:
    text = phrase
    text = re.sub(r"^(a|an|the)\s+", "", text)
    text = re.sub(r"\b(is|are|was|were)\b", "", text, count=1)
    text = re.sub(r"\s+", " ", text).strip()
    return text or phrase


def build_replacement_pool(original: str) -> list[str]:
    base = normalize_text(original)
    phrase = make_phrase(original)
    compact = compress_phrase(phrase)
    articleless = re.sub(r"^(a|an|the)\s+", "", phrase).strip() or phrase
    compact_no_article = re.sub(r"^(a|an|the)\s+", "", compact).strip() or compact

    candidates = [
        base,
        compact,
        phrase,
        articleless,
        compact_no_article,
        f"video of {compact_no_article}",
        f"clip of {compact_no_article}",
        f"scene of {compact_no_article}",
        f"scene with {compact_no_article}",
        f"showing {compact_no_article}",
        f"featuring {compact_no_article}",
        f"footage of {compact_no_article}",
        f"view of {compact_no_article}",
        f"video showing {compact_no_article}",
        f"clip showing {compact_no_article}",
        f"scene showing {compact_no_article}",
        f"{compact_no_article} on screen",
        f"{compact_no_article} in the clip",
        f"search for {compact_no_article}",
        f"video search for {compact_no_article}",
    ]

    pool: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        norm = normalize_text(item)
        if not norm:
            continue
        if words(norm) and words(norm)[-1] in BAD_ENDINGS:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        pool.append(norm)
    return pool


def choose_replacement(original: str, used: set[str], slot: int) -> str:
    pool = build_replacement_pool(original)
    for candidate in pool:
        if candidate not in used:
            used.add(candidate)
            return candidate

    fallback_base = normalize_text(original)
    idx = slot + 1
    while True:
        candidate = f"{fallback_base} variation {idx}"
        candidate = normalize_text(candidate)
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--replacements-jsonl", type=Path, default=DEFAULT_REPLACEMENTS_JSONL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_json = args.input_json if args.in_place else args.output_json

    if args.overwrite:
        paths = [args.report_json, args.replacements_jsonl]
        if not args.in_place:
            paths.append(output_json)
        for path in paths:
            path.unlink(missing_ok=True)

    with args.source_json.open() as f:
        source = json.load(f)
    with args.input_json.open() as f:
        addition = json.load(f)

    source_by_video = {item["video"]: item["caption"] for item in source}
    grouped = defaultdict(list)
    for idx, item in enumerate(addition):
        grouped[item["video"]].append((idx, item["caption"]))

    repaired = [None] * len(addition)
    replacement_logs: list[dict[str, object]] = []
    reason_counter: Counter[str] = Counter()
    replaced_queries = 0
    replaced_videos = 0

    for video, items in grouped.items():
        original = source_by_video[video]
        flags = [flag_query(caption, original) for _, caption in items]
        video_flagged = any(flag.flagged for flag in flags)

        if video_flagged:
            used: set[str] = set()
            for flag in flags:
                for reason in flag.reasons:
                    reason_counter[reason] += 1

            for slot, ((idx, caption), flag) in enumerate(zip(items, flags)):
                replacement = choose_replacement(original, used, slot)
                repaired[idx] = {"video": video, "caption": replacement}
                if normalize_text(caption) != replacement:
                    replaced_queries += 1
                    replacement_logs.append(
                        {
                            "video": video,
                            "index": idx,
                            "old_caption": caption,
                            "new_caption": replacement,
                            "reasons": flag.reasons if flag.reasons else ["same_video_rewrite"],
                        }
                    )
            replaced_videos += 1
        else:
            for idx, caption in items:
                repaired[idx] = {"video": video, "caption": caption}

    assert all(item is not None for item in repaired)

    tmp_output = output_json.with_suffix(output_json.suffix + ".tmp")
    with tmp_output.open("w") as f:
        json.dump(repaired, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_output, output_json)

    args.replacements_jsonl.unlink(missing_ok=True)
    append_jsonl(args.replacements_jsonl, replacement_logs)

    report = {
        "input_entries": len(addition),
        "output_entries": len(repaired),
        "replaced_queries": replaced_queries,
        "replaced_videos": replaced_videos,
        "reason_counts": dict(sorted(reason_counter.items())),
        "output_path": str(output_json),
    }
    tmp_report = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
    with tmp_report.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    os.replace(tmp_report, args.report_json)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
