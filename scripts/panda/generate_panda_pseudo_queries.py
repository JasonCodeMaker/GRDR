#!/usr/bin/env python3
"""Generate diverse pseudo queries for Panda-70M-10M train clips."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import StoppingCriteriaList


GRDR_ROOT = Path(__file__).resolve().parent.parent
CAPTION_ROOT = Path("/data2/uqzzha35/VideoRetrieval/Panda-70M-10M/captioning")
DEFAULT_FRAMES_ROOT = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "panda_10m_frames" / "train"
DEFAULT_SOURCE_JSON = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption" / "panda_10m_ret_train.json"
DEFAULT_OUTPUT_JSON = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption" / "panda_10m_ret_train_addition.json"
DEFAULT_PROGRESS_JSONL = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption" / "panda_10m_ret_train_addition.progress.jsonl"
DEFAULT_FAILURES_JSONL = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption" / "panda_10m_ret_train_addition.failures.jsonl"
DEFAULT_SUMMARY_JSON = GRDR_ROOT / "dataset" / "Panda-70M-10M" / "video_retreival_caption" / "panda_10m_ret_train_addition.summary.json"

PROMPT_GROUPS = [
    {
        "style": "main_action",
        "prompt": (
            "Write one complete sentence that begins with 'A' or 'An' and describes the main visible action "
            "and scene in this clip. Use only visible content. Do not infer names, channels, locations, or "
            "story context unless readable text clearly shows them."
        ),
        "temperature": 1.08,
        "top_p": 0.88,
        "num_beams": 4,
        "num_return_sequences": 2,
    },
    {
        "style": "people_focus",
        "prompt": (
            "Write one complete sentence that begins with 'A', 'An', 'Two', or 'Three' and focuses on the "
            "visible person or people, what they are doing, and any clearly visible relation or clothing detail. "
            "Use only visible content."
        ),
        "temperature": 1.14,
        "top_p": 0.89,
        "num_beams": 4,
        "num_return_sequences": 2,
    },
    {
        "style": "object_scene",
        "prompt": (
            "Write one complete sentence that begins with 'A' or 'An' and focuses on the visible objects, "
            "setting, colors, or layout in this clip. Use only visible content."
        ),
        "temperature": 1.16,
        "top_p": 0.90,
        "num_beams": 4,
        "num_return_sequences": 2,
    },
    {
        "style": "screen_text",
        "prompt": (
            "If the clip prominently shows a screen, sign, title card, interface, or readable text, write one "
            "complete sentence describing that visible screen or text. Otherwise write one sentence about the most "
            "salient visible object or scene."
        ),
        "temperature": 1.04,
        "top_p": 0.86,
        "num_beams": 3,
        "num_return_sequences": 1,
    },
    {
        "style": "there_is",
        "prompt": (
            "Write one complete sentence that begins with 'There is' or 'There are' and describes the most "
            "salient visible scene, arrangement, or background. Use only visible content."
        ),
        "temperature": 1.06,
        "top_p": 0.86,
        "num_beams": 3,
        "num_return_sequences": 1,
    },
    {
        "style": "talking_head",
        "prompt": (
            "If a speaker, presenter, news anchor, interviewee, or talking-head is visible, write one complete "
            "sentence in the style of 'The speaker ...' or 'A news anchor ...'. Otherwise write one complete "
            "sentence about the visible person and background. Use only visible content."
        ),
        "temperature": 1.10,
        "top_p": 0.88,
        "num_beams": 3,
        "num_return_sequences": 1,
    },
]

FALLBACK_GROUPS = [
    {
        "style": "generic_caption",
        "prompt": (
            "Write one complete sentence in the style of a video dataset caption, similar to 'A person is ...' "
            "or 'There is ...'. Use only visible content and keep the sentence grounded in what can be seen."
        ),
        "temperature": 1.18,
        "top_p": 0.90,
        "num_beams": 4,
        "num_return_sequences": 2,
    },
    {
        "style": "alt_caption",
        "prompt": (
            "Write one more complete sentence for this clip with slightly different emphasis, but keep the same "
            "caption style and use only visible content."
        ),
        "temperature": 1.24,
        "top_p": 0.91,
        "num_beams": 4,
        "num_return_sequences": 2,
    },
]

BAD_ENDINGS = {
    "and",
    "or",
    "with",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "the",
    "a",
    "an",
    "one",
    "their",
    "his",
    "her",
    "there",
    "is",
    "are",
    "front",
    "back",
    "he",
    "she",
    "they",
    "that",
    "this",
    "those",
    "these",
    "something",
}
CONTENT_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "being",
    "been",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "as",
    "into",
    "onto",
    "over",
    "under",
    "through",
    "around",
    "while",
    "there",
    "it",
    "its",
    "their",
    "his",
    "her",
    "he",
    "she",
    "they",
    "them",
    "that",
    "this",
    "these",
    "those",
    "someone",
    "somebody",
}
MIN_WORDS = 4
MAX_WORDS = 32
SIMILARITY_THRESHOLD = 0.82
SCREEN_TEXT_PATTERNS = [
    r"\bscreen\b",
    r"\btv\b",
    r"\btelevision\b",
    r"\btitle screen\b",
    r"\bgame over\b",
    r"\binterface\b",
    r"\bmenu\b",
    r"\bwebsite\b",
    r"\bweb page\b",
    r"\bpage\b",
    r"\blogo\b",
    r"\bsign\b",
    r"\bposter\b",
    r"\bcaption\b",
    r"\bsubtitle\b",
    r"\btext\b",
    r"\bthe words\b",
    r"\breads\b",
    r"\bphone\b",
    r"\btablet\b",
]
TALKING_HEAD_PATTERNS = [
    r"\bspeaker\b",
    r"\banchor\b",
    r"\bpresenter\b",
    r"\breporter\b",
    r"\binterview\b",
    r"\binterviewee\b",
    r"\bnews\b",
    r"\bnewscast\b",
    r"\bbroadcast\b",
    r"\btalking to the camera\b",
    r"\bspeaking to the camera\b",
    r"\bon a television screen\b",
    r"\bon a tv screen\b",
    r"\bin a news studio\b",
]
PLURAL_HINT_PATTERNS = [
    r"^two\b",
    r"^three\b",
    r"^four\b",
    r"^several\b",
    r"^many\b",
    r"^people\b",
    r"^men\b",
    r"^women\b",
    r"^children\b",
    r"^a group of\b",
]


def clean_caption_text(text: str) -> str:
    text = text.strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.?!;:,]+$", "", text)
    return text.strip()


def normalize_caption(text: str) -> str:
    return clean_caption_text(text).lower()


def format_caption_like_panda(text: str) -> str:
    text = clean_caption_text(text)
    if not text:
        return ""
    if text[0].isalpha() and text[0].islower():
        text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


def content_words(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9']+", normalize_caption(text))
        if token not in CONTENT_STOPWORDS
    }


def has_reference_overlap(candidate: str, reference: str) -> bool:
    ref_words = content_words(reference)
    if not ref_words:
        return True
    return bool(content_words(candidate) & ref_words)


def jaccard_similarity(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r"[a-z0-9']+", normalize_caption(text_a)))
    words_b = set(re.findall(r"[a-z0-9']+", normalize_caption(text_b)))
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def is_valid_candidate(candidate: str, reference: str) -> bool:
    if not candidate:
        return False
    words = re.findall(r"[a-z0-9']+", normalize_caption(candidate))
    if len(words) < MIN_WORDS or len(words) > MAX_WORDS:
        return False
    if words and words[-1] in BAD_ENDINGS:
        return False
    if normalize_caption(candidate) == "video content":
        return False
    return has_reference_overlap(candidate, reference)


def reference_matches(reference: str, patterns: list[str]) -> bool:
    ref = normalize_caption(reference)
    return any(re.search(pattern, ref) for pattern in patterns)


def style_applies(style: str, reference: str) -> bool:
    if style == "screen_text":
        return reference_matches(reference, SCREEN_TEXT_PATTERNS)
    if style == "talking_head":
        return reference_matches(reference, TALKING_HEAD_PATTERNS)
    return True


def strip_leading_article(text: str) -> str:
    return re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE).strip()


def strip_group_prefix(text: str) -> str:
    text = strip_leading_article(text)
    text = re.sub(r"^group of\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def build_people_subject(subject: str) -> str:
    subject = strip_group_prefix(subject)
    subject = re.sub(
        r"^(young|old|older|adult|little|small)\s+(men|women|boys|girls|children|people)\b",
        r"people",
        subject,
        flags=re.IGNORECASE,
    )
    subject = re.sub(r"^(men|women|boys|girls|children)\b", "people", subject, flags=re.IGNORECASE)
    return subject.strip()


def parse_reference_caption(reference: str) -> dict[str, str]:
    core = clean_caption_text(reference)
    lower = normalize_caption(reference)
    match = re.match(
        r"^(a|an|the|two|three|four|several|many|people|men|women|children)\s+(.+?)\s+\b(is|are|was|were)\b\s+(.+)$",
        lower,
    )
    if match:
        determiner, subject, copula, predicate = match.groups()
        return {
            "core": core,
            "lower": lower,
            "subject_with_det": f"{determiner} {subject}".strip(),
            "subject": subject.strip(),
            "copula": copula,
            "predicate": predicate.strip(),
        }
    return {
        "core": core,
        "lower": lower,
        "subject_with_det": "",
        "subject": "",
        "copula": "",
        "predicate": "",
    }


def is_plural_reference(reference: str, parsed: dict[str, str]) -> bool:
    lower = parsed["lower"] or normalize_caption(reference)
    return any(re.search(pattern, lower) for pattern in PLURAL_HINT_PATTERNS)


def build_style_fallbacks(reference: str, style: str) -> list[str]:
    base = format_caption_like_panda(reference)
    if not base:
        return ["A video clip is shown."]

    parsed = parse_reference_caption(reference)
    core = parsed["core"]
    subject = parsed["subject"]
    predicate = parsed["predicate"]
    bare_subject = strip_group_prefix(parsed["subject_with_det"] or subject or core)
    lower_core = core[0].lower() + core[1:] if core and core[0].isalpha() else core
    plural = is_plural_reference(reference, parsed)

    variants: list[str] = []
    if style == "main_action":
        variants.append(base)
    elif style == "people_focus":
        if bare_subject and predicate:
            variants.append(format_caption_like_panda(f"{bare_subject.capitalize()} {parsed['copula']} {predicate}"))
        people_subject = build_people_subject(parsed["subject_with_det"] or subject or core)
        if people_subject and predicate:
            variants.append(format_caption_like_panda(f"{people_subject.capitalize()} {parsed['copula']} {predicate}"))
        variants.append(base)
    elif style == "object_scene":
        location_match = re.search(
            r"\b(?:in|on|at|inside|outside|near|by|along|through|down)\s+(a|an|the)\s+([a-z0-9' -]+)$",
            parsed["lower"],
        )
        if location_match and bare_subject and predicate:
            article, location = location_match.groups()
            location_phrase = f"{article} {location}".strip()
            scene_label = strip_leading_article(location_phrase)
            scene_predicate = re.sub(re.escape(location_phrase), "it", predicate, flags=re.IGNORECASE)
            variants.append(
                format_caption_like_panda(
                    f"A {scene_label} scene with {bare_subject} {scene_predicate}"
                )
            )
        if bare_subject and predicate:
            variants.append(format_caption_like_panda(f"A scene with {bare_subject} {predicate}"))
        variants.append(base)
    elif style == "screen_text":
        variants.append(base)
        if "words" in parsed["lower"] or "text" in parsed["lower"] or "reads" in parsed["lower"]:
            variants.append(format_caption_like_panda(f"A screen showing {lower_core}"))
    elif style == "there_is":
        prefix = "There are" if plural else "There is"
        if bare_subject and predicate:
            variants.append(format_caption_like_panda(f"{prefix} {bare_subject} {predicate}"))
        else:
            variants.append(format_caption_like_panda(f"{prefix} {lower_core}"))
        variants.append(base)
    elif style == "talking_head":
        if reference_matches(reference, TALKING_HEAD_PATTERNS):
            if re.search(r"\bnews\b|\banchor\b|\breporter\b|\bnewscast\b|\bbroadcast\b", parsed["lower"]):
                variants.append(format_caption_like_panda(f"A news anchor is {lower_core}"))
            variants.append(format_caption_like_panda(f"The speaker is {lower_core}"))
        variants.append(base)
    else:
        variants.append(base)

    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = canonical_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_reference_fallbacks(reference: str) -> list[str]:
    base = format_caption_like_panda(reference)
    if not base:
        return ["A video clip is shown."]

    styles = ["main_action", "people_focus", "object_scene", "there_is"]
    if style_applies("screen_text", reference):
        styles.append("screen_text")
    if style_applies("talking_head", reference):
        styles.append("talking_head")

    variants: list[str] = []
    for style in styles:
        variants.extend(build_style_fallbacks(reference, style))

    out: list[str] = []
    seen: set[str] = set()
    for item in variants:
        key = canonical_key(item)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def canonical_key(text: str) -> str:
    return normalize_caption(text)


def load_source_records(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        records = json.load(f)
    cleaned = []
    for item in records:
        video = str(item["video"]).strip()
        caption = str(item.get("caption", "")).strip()
        cleaned.append({"video": video, "caption": caption})
    return cleaned


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done.add(rec["video"])
    return done


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class PandaFrameDataset(Dataset):
    def __init__(self, records: list[dict[str, str]], frames_root: Path, transform):
        self.records = records
        self.frames_root = Path(frames_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        video = self.records[idx]["video"]
        clip_id = Path(video).stem
        clip_dir = self.frames_root / clip_id
        try:
            frame_files = sorted(clip_dir.glob("frame_*.jpg"))
            if not frame_files:
                raise RuntimeError(f"no frames found in {clip_dir}")
            frames = [np.asarray(Image.open(frame).convert("RGB")) for frame in frame_files]
            video_tensor = torch.from_numpy(np.stack(frames)).permute(3, 0, 1, 2).float()
            video_tensor = self.transform(video_tensor)
            return {"video": video, "frames": video_tensor, "error": ""}
        except Exception as exc:  # pragma: no cover - defensive runtime path
            return {"video": video, "frames": None, "error": f"{type(exc).__name__}: {exc}"}


def collate_panda_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    valid = [item for item in batch if item["frames"] is not None]
    invalid = [{"video": item["video"], "error": item["error"]} for item in batch if item["frames"] is None]
    frames = None
    videos: list[str] = []
    if valid:
        frames = torch.stack([item["frames"] for item in valid], dim=0)
        videos = [str(item["video"]) for item in valid]
    return {"frames": frames, "videos": videos, "invalid": invalid}


def load_model_and_processor():
    sys.path.insert(0, str(CAPTION_ROOT))
    from video_llama.common.config import Config
    from video_llama.common.registry import registry
    from video_llama.models.video_llama import StoppingCriteriaSub
    import video_llama.tasks  # noqa: F401

    class Args:
        cfg_path = str(CAPTION_ROOT / "eval_configs" / "panda70M_eval.yaml")
        options = None

    cfg = Config(Args())
    model_cfg = cfg.model_cfg
    model_cfg.llama_model = str(CAPTION_ROOT / model_cfg.llama_model)
    model_cfg.ckpt = str(CAPTION_ROOT / model_cfg.ckpt)
    model = registry.get_model_class(model_cfg.arch).from_config(model_cfg).to("cuda").eval()

    vis_processor_cfg = DotDict({"name": "alpro_video_eval", "n_frms": 8, "image_size": 224})
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [torch.tensor([2], device="cuda")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return model, vis_processor, stopping_criteria


@torch.inference_mode()
def generate_group(
    model,
    frames: torch.Tensor,
    prompts: list[str],
    stopping_criteria: StoppingCriteriaList,
    *,
    num_beams: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
    max_new_tokens: int,
) -> list[list[str]]:
    img_embeds, img_atts = model.encode_videoQformer_visual(frames)

    batch_size = img_embeds.shape[0]
    bos = torch.full(
        (batch_size, 1),
        model.llama_tokenizer.bos_token_id,
        dtype=torch.long,
        device=frames.device,
    )
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    bos_atts = img_atts[:, :1]

    txt_embeds, txt_atts = model.encode_textQformer_prompt(prompts, img_embeds)
    mixed_embeds = torch.cat([bos_embeds, txt_embeds, img_embeds], dim=1)
    mixed_atts = torch.cat([bos_atts, txt_atts, img_atts], dim=1)

    outputs = model.llama_model.generate(
        inputs_embeds=mixed_embeds,
        attention_mask=mixed_atts,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=num_beams,
        do_sample=True,
        min_length=5,
        top_p=top_p,
        repetition_penalty=1.10,
        length_penalty=1.0,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        use_cache=True,
    )

    decoded: list[list[str]] = [[] for _ in range(batch_size)]
    for sample_idx in range(batch_size):
        for seq_idx in range(num_return_sequences):
            row = sample_idx * num_return_sequences + seq_idx
            output_token = outputs[row]
            if output_token[0] == model.llama_tokenizer.bos_token_id:
                output_token = output_token[1:]
            text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
            text = text.split(model.end_sym)[0].strip()
            cleaned = format_caption_like_panda(text)
            if cleaned:
                decoded[sample_idx].append(cleaned)
    return decoded


def build_unique_queries(
    model,
    frames: torch.Tensor,
    videos: list[str],
    source_captions: dict[str, str],
    stopping_criteria: StoppingCriteriaList,
    queries_per_video: int,
) -> dict[str, list[str]]:
    results = {video: [] for video in videos}
    seen = {video: set() for video in videos}

    def maybe_add(video: str, text: str) -> None:
        formatted = format_caption_like_panda(text)
        key = canonical_key(formatted)
        if not key or key in seen[video]:
            return
        if not is_valid_candidate(formatted, source_captions.get(video, "")):
            return
        if any(jaccard_similarity(formatted, existing) >= SIMILARITY_THRESHOLD for existing in results[video]):
            return
        results[video].append(formatted)
        seen[video].add(key)

    for group in PROMPT_GROUPS:
        applicable = [video for video in videos if len(results[video]) < queries_per_video and style_applies(group["style"], source_captions.get(video, ""))]
        if not applicable:
            continue
        positions = {video: idx for idx, video in enumerate(videos)}
        indices = [positions[video] for video in applicable]
        sub_frames = frames[indices]
        prompts = [group["prompt"]] * len(applicable)
        before_counts = {video: len(results[video]) for video in applicable}
        group_outputs = generate_group(
            model,
            sub_frames,
            prompts,
            stopping_criteria,
            num_beams=group["num_beams"],
            temperature=group["temperature"],
            top_p=group["top_p"],
            num_return_sequences=group["num_return_sequences"],
            max_new_tokens=32,
        )
        for video, outputs in zip(applicable, group_outputs):
            for text in outputs:
                maybe_add(video, text)
                if len(results[video]) >= queries_per_video:
                    break
            if len(results[video]) == before_counts[video]:
                for text in build_style_fallbacks(source_captions.get(video, ""), group["style"]):
                    maybe_add(video, text)
                    if len(results[video]) >= queries_per_video:
                        break

    if all(len(results[video]) >= queries_per_video for video in videos):
        return {video: results[video][:queries_per_video] for video in videos}

    for group in FALLBACK_GROUPS:
        unfinished = [video for video in videos if len(results[video]) < queries_per_video]
        if not unfinished:
            break
        positions = {video: idx for idx, video in enumerate(videos)}
        indices = [positions[video] for video in unfinished]
        sub_frames = frames[indices]
        prompts = [group["prompt"]] * len(unfinished)
        group_outputs = generate_group(
            model,
            sub_frames,
            prompts,
            stopping_criteria,
            num_beams=group["num_beams"],
            temperature=group["temperature"],
            top_p=group["top_p"],
            num_return_sequences=group["num_return_sequences"],
            max_new_tokens=32,
        )
        for video, outputs in zip(unfinished, group_outputs):
            for text in outputs:
                maybe_add(video, text)
                if len(results[video]) >= queries_per_video:
                    break

    for video in videos:
        for text in build_reference_fallbacks(source_captions.get(video, "")):
            maybe_add(video, text)
            if len(results[video]) >= queries_per_video:
                break
        if not results[video]:
            results[video] = [format_caption_like_panda(source_captions.get(video, "")) or "A video clip is shown."]
        while len(results[video]) < queries_per_video:
            results[video].append(results[video][len(results[video]) % len(results[video])])
        results[video] = results[video][:queries_per_video]

    return results


def append_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for _ in f)


def finalize_json(progress_path: Path, output_path: Path) -> int:
    total_entries = 0
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with progress_path.open() as src, tmp_path.open("w") as dst:
        dst.write("[\n")
        first = True
        for line in src:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            video = rec["video"]
            for caption in rec["captions"]:
                row = {"video": video, "caption": caption}
                if not first:
                    dst.write(",\n")
                dst.write(json.dumps(row, ensure_ascii=False, indent=2))
                first = False
                total_entries += 1
        dst.write("\n]\n")
    os.replace(tmp_path, output_path)
    return total_entries


def write_summary(path: Path, *, total_videos: int, output_entries: int, failures: int, queries_per_video: int) -> None:
    payload = {
        "total_videos": total_videos,
        "queries_per_video": queries_per_video,
        "output_entries": output_entries,
        "failure_videos": failures,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", type=Path, default=DEFAULT_FRAMES_ROOT)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--progress-jsonl", type=Path, default=DEFAULT_PROGRESS_JSONL)
    parser.add_argument("--failures-jsonl", type=Path, default=DEFAULT_FAILURES_JSONL)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--queries-per-video", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.overwrite:
        for path in (args.output_json, args.progress_jsonl, args.failures_jsonl, args.summary_json):
            path.unlink(missing_ok=True)

    records = load_source_records(args.source_json)
    if args.max_videos is not None:
        records = records[: args.max_videos]
    source_captions = {record["video"]: record.get("caption", "") for record in records}

    done = load_done(args.progress_jsonl)
    todo = [record for record in records if record["video"] not in done]
    print(f"[data] total={len(records)} done={len(done)} todo={len(todo)}")

    if todo:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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

        done_count = len(done)
        pbar = tqdm(loader, desc="pseudo-queries")
        for batch in pbar:
            if batch["invalid"]:
                append_jsonl(args.failures_jsonl, batch["invalid"])

            frames = batch["frames"]
            videos = batch["videos"]
            if frames is None or not videos:
                continue

            frames = frames.to("cuda", non_blocking=True)
            generated = build_unique_queries(
                model,
                frames,
                videos,
                source_captions,
                stopping_criteria,
                queries_per_video=args.queries_per_video,
            )
            append_records = [{"video": video, "captions": generated[video]} for video in videos]
            append_jsonl(args.progress_jsonl, append_records)
            done_count += len(videos)
            pbar.set_postfix(done=done_count)

    output_entries = finalize_json(args.progress_jsonl, args.output_json)
    failures = count_jsonl_lines(args.failures_jsonl)
    write_summary(
        args.summary_json,
        total_videos=count_jsonl_lines(args.progress_jsonl),
        output_entries=output_entries,
        failures=failures,
        queries_per_video=args.queries_per_video,
    )
    print(f"[done] wrote {output_entries} entries -> {args.output_json}")


if __name__ == "__main__":
    main()
