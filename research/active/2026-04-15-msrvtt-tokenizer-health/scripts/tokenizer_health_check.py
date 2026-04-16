#!/usr/bin/env python3
"""Standalone tokenizer health check for GRDR VideoRQVAE checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.data_utils import load_shared_features  # noqa: E402
from utils.model_utils import create_videorqvae  # noqa: E402
from utils.training_utils import safe_load  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to best_model.pt")
    parser.add_argument("--dataset", default="msrvtt", help="Dataset name for feature loading")
    parser.add_argument("--features_root", default="dataset/features", help="Root directory for video/text features")
    parser.add_argument("--code_num", type=int, default=128, help="Codebook size per RQ layer")
    parser.add_argument("--code_length", type=int, default=3, help="Number of RQ layers")
    parser.add_argument("--num_latent_tokens", type=int, default=4, help="Number of latent tokens")
    parser.add_argument("--embed_dim", type=int, default=512, help="VideoRQVAE embedding dimension")
    parser.add_argument("--in_dim", type=int, default=512, help="Input feature dimension")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for tokenizer analysis")
    parser.add_argument("--device", default=None, help="Torch device, defaults to cuda if available")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both", help="Which split(s) to analyze")
    parser.add_argument("--json_out", default=None, help="Optional path to write the JSON report")
    parser.add_argument("--quiet", action="store_true", help="Silence feature-loading logs")
    return parser.parse_args()


def offdiag_stats(matrix: np.ndarray, pair_idx: List[Tuple[int, int]]) -> Dict[str, object]:
    values = [float(matrix[i, j]) for i, j in pair_idx]
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "pairs": {f"{i}-{j}": round(float(matrix[i, j]), 4) for i, j in pair_idx},
    }


def counter_stats(counter: Counter, denom: int) -> Dict[str, object]:
    top_code, top_count = counter.most_common(1)[0]
    probs = np.array(list(counter.values()), dtype=np.float64) / float(denom)
    return {
        "active_codes": len(counter),
        "top_code": int(top_code) if not isinstance(top_code, tuple) else list(top_code),
        "top_share": top_count / float(denom),
        "entropy_bits": float(-(probs * np.log2(probs)).sum()),
    }


def analyze_split(
    model,
    video_dict: Dict[str, np.ndarray],
    num_latent_tokens: int,
    code_length: int,
    batch_size: int,
    device: str,
) -> Dict[str, object]:
    keys = list(video_dict.keys())
    feats = np.stack([video_dict[k] for k in keys]).astype("float32")
    pair_idx = [(i, j) for i in range(num_latent_tokens) for j in range(i + 1, num_latent_tokens)]

    enc_sim_sum = np.zeros((num_latent_tokens, num_latent_tokens), dtype=np.float64)
    quant_sim_sum = np.zeros((num_latent_tokens, num_latent_tokens), dtype=np.float64)
    total_videos = 0
    pair_total = 0

    any_dup_last = 0
    any_dup_full = 0
    dup_pair_last = 0
    dup_pair_full = 0
    unique_last_sum = 0
    unique_full_sum = 0
    recon_cos_sum = 0.0

    slot_last_counters = [Counter() for _ in range(num_latent_tokens)]
    global_last_counter = Counter()

    layer_video_dup = [0] * code_length
    layer_pair_dup = [0] * code_length
    layer_global = [Counter() for _ in range(code_length)]
    layer_slot = [[Counter() for _ in range(num_latent_tokens)] for _ in range(code_length)]

    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            batch_np = feats[start:start + batch_size]
            x = torch.from_numpy(batch_np).to(device)

            x_encoded = model.encoder(x)
            q_video_emb, _, indices, _, _ = model.rq(x_encoded, use_sk=False, return_probs=False)
            _, reconstructed = model.decoder(q_video_emb)

            x_enc_norm = F.normalize(x_encoded, dim=-1, eps=1e-12)
            q_norm = F.normalize(q_video_emb, dim=-1, eps=1e-12)
            enc_sim = torch.matmul(x_enc_norm, x_enc_norm.transpose(1, 2)).cpu().numpy()
            quant_sim = torch.matmul(q_norm, q_norm.transpose(1, 2)).cpu().numpy()
            enc_sim_sum += enc_sim.sum(axis=0)
            quant_sim_sum += quant_sim.sum(axis=0)

            recon_cos = F.cosine_similarity(
                F.normalize(reconstructed, dim=-1, eps=1e-12),
                F.normalize(x, dim=-1, eps=1e-12),
                dim=-1,
            )
            recon_cos_sum += recon_cos.sum().item()

            idx_np = indices.cpu().numpy()
            total_videos += idx_np.shape[0]
            pair_total += idx_np.shape[0] * len(pair_idx)

            for sample in idx_np:
                last_codes = [int(v[-1]) for v in sample]
                full_codes = [tuple(int(x) for x in v) for v in sample]

                unique_last = len(set(last_codes))
                unique_full = len(set(full_codes))
                unique_last_sum += unique_last
                unique_full_sum += unique_full
                any_dup_last += int(unique_last < num_latent_tokens)
                any_dup_full += int(unique_full < num_latent_tokens)

                for i, j in pair_idx:
                    dup_pair_last += int(last_codes[i] == last_codes[j])
                    dup_pair_full += int(full_codes[i] == full_codes[j])

                for slot in range(num_latent_tokens):
                    slot_last_counters[slot][last_codes[slot]] += 1
                    global_last_counter[last_codes[slot]] += 1

                for layer in range(code_length):
                    layer_codes = [int(v[layer]) for v in sample]
                    if len(set(layer_codes)) < num_latent_tokens:
                        layer_video_dup[layer] += 1
                    for slot in range(num_latent_tokens):
                        layer_slot[layer][slot][layer_codes[slot]] += 1
                        layer_global[layer][layer_codes[slot]] += 1
                    for i, j in pair_idx:
                        layer_pair_dup[layer] += int(layer_codes[i] == layer_codes[j])

    enc_sim_mean = enc_sim_sum / total_videos
    quant_sim_mean = quant_sim_sum / total_videos

    return {
        "num_videos": total_videos,
        "recon_cos_mean": recon_cos_sum / float(total_videos),
        "encoder_pairwise_cosine": offdiag_stats(enc_sim_mean, pair_idx),
        "quantized_pairwise_cosine": offdiag_stats(quant_sim_mean, pair_idx),
        "duplicate_last_code_video_rate": any_dup_last / float(total_videos),
        "duplicate_full_code_video_rate": any_dup_full / float(total_videos),
        "duplicate_last_code_pair_rate": dup_pair_last / float(pair_total),
        "duplicate_full_code_pair_rate": dup_pair_full / float(pair_total),
        "mean_unique_last_codes_per_video": unique_last_sum / float(total_videos),
        "mean_unique_full_codes_per_video": unique_full_sum / float(total_videos),
        "global_last_code_stats": counter_stats(global_last_counter, total_videos * num_latent_tokens),
        "slot_last_code_stats": {
            str(slot): counter_stats(slot_last_counters[slot], total_videos)
            for slot in range(num_latent_tokens)
        },
        "layer_breakdown": {
            str(layer): {
                "video_duplicate_rate": layer_video_dup[layer] / float(total_videos),
                "pair_duplicate_rate": layer_pair_dup[layer] / float(pair_total),
                "global_active_codes": len(layer_global[layer]),
                "global_top_share": layer_global[layer].most_common(1)[0][1] / float(total_videos * num_latent_tokens),
                "slot_active_codes": [len(layer_slot[layer][slot]) for slot in range(num_latent_tokens)],
                "slot_top_share": [
                    layer_slot[layer][slot].most_common(1)[0][1] / float(total_videos)
                    for slot in range(num_latent_tokens)
                ],
            }
            for layer in range(code_length)
        },
    }


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log_fn = (lambda *_args, **_kwargs: None) if args.quiet else print

    checkpoint = Path(args.checkpoint)
    videorqvae_path = checkpoint.with_name(checkpoint.name + ".videorqvae")
    if not videorqvae_path.exists():
        raise FileNotFoundError(f"Missing VideoRQVAE checkpoint: {videorqvae_path}")

    feature_cache = load_shared_features(
        args.dataset,
        args.features_root,
        log_fn,
        use_pseudo_queries=False,
    )

    model = create_videorqvae(
        code_num=args.code_num,
        code_length=args.code_length,
        num_latent_tokens=args.num_latent_tokens,
        e_dim=args.embed_dim,
        in_dim=args.in_dim,
        device=device,
    )
    safe_load(model, str(videorqvae_path))
    model.eval()

    result = {
        "checkpoint": str(checkpoint),
        "videorqvae_checkpoint": str(videorqvae_path),
        "device": device,
        "config": {
            "dataset": args.dataset,
            "code_num": args.code_num,
            "code_length": args.code_length,
            "num_latent_tokens": args.num_latent_tokens,
            "embed_dim": args.embed_dim,
            "in_dim": args.in_dim,
            "batch_size": args.batch_size,
        },
    }

    if args.split in ("train", "both"):
        result["train"] = analyze_split(
            model,
            feature_cache["train_video"],
            args.num_latent_tokens,
            args.code_length,
            args.batch_size,
            device,
        )
    if args.split in ("test", "both"):
        result["test"] = analyze_split(
            model,
            feature_cache["test_video"],
            args.num_latent_tokens,
            args.code_length,
            args.batch_size,
            device,
        )

    payload = json.dumps(result, indent=2)
    if args.json_out is not None:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
