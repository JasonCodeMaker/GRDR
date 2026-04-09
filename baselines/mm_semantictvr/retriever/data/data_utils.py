import json
from pathlib import Path

import torch

from utils.data_utils import load_or_compute_kmeans_cache


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = REPO_ROOT / "data"


def _dataset_dir(dataset_name: str) -> str:
    if dataset_name in {"actnet", "activitynet"}:
        return "actnet"
    return dataset_name


def _caption_prefix(dataset_name: str) -> str:
    if dataset_name in {"actnet", "activitynet"}:
        return "actnet"
    return dataset_name


def load_caption_annotations(dataset_name, split="train"):
    dataset_dir = _dataset_dir(dataset_name)
    caption_prefix = _caption_prefix(dataset_name)
    caption_file = DATA_ROOT / dataset_dir / "video_retreival_caption" / f"{caption_prefix}_ret_{split}.json"

    if dataset_dir == "lsmdc" and split == "test" and not caption_file.exists():
        caption_file = DATA_ROOT / dataset_dir / "video_retreival_caption" / f"{caption_prefix}_ret_test_1000.json"

    with open(caption_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def indices_to_string(token_idx, codes, code_num, codebook_layers):
    def get_prefix(idx):
        if idx < 26:
            return chr(ord("a") + idx)
        first = idx // 26 - 1
        second = idx % 26
        return chr(ord("a") + first) + chr(ord("a") + second)

    codes_list = codes.tolist() if isinstance(codes, torch.Tensor) else codes
    return " ".join(f"{get_prefix(layer_idx)}_{code_idx}" for layer_idx, code_idx in enumerate(codes_list))
