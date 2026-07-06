import os
import json
import argparse
import csv
import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import CLIPTokenizer

from model.clip_transformer import CLIPTransformer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_file', required=True)
    parser.add_argument('--eval_checkpoint', required=True)
    parser.add_argument('--video_cache_dir', required=True)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_json')
    parser.add_argument('--result_file',
                        help='CSV result path, relative to output/evaluation_results/rerank unless absolute')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--load_workers', type=int, default=16,
                        help='threads for parallel np.load of candidate frame embeds')
    parser.add_argument('--max_candidates', type=int, default=0,
                        help='cap each query to its top-K stage-1 candidates before rerank (0 = no cap)')
    args = parser.parse_args()
    if not args.out_json and not args.result_file:
        parser.error('at least one of --out_json or --result_file is required')
    return args


def set_seed(seed):
    """Seed RNGs to mirror test.py determinism."""
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(eval_checkpoint, device):
    """Build CLIPTransformer (HF, transformer pooling) and load the X-Pool state_dict exactly as test.py does."""
    config = SimpleNamespace(huggingface=True, pooling_type='transformer', embed_dim=512,
                             num_mha_heads=1, transformer_dropout=0.3)
    model = CLIPTransformer(config)
    checkpoint = torch.load(eval_checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device).eval()
    return model


def load_frame_cache(video_id, cache_dir, num_frames):
    """Load cached [num_frames, 512] frame_embeds for a bare id from PANDA test/ or train/."""
    panda_dir = os.path.join(cache_dir, 'PANDA')
    for split in ('test', 'train'):
        path = os.path.join(panda_dir, split, f"{video_id}.npz")
        if os.path.exists(path):
            frame_embeds = np.load(path)['frame_embeds']
            assert frame_embeds.shape == (num_frames, 512), \
                f"{video_id}: got {frame_embeds.shape}, expected ({num_frames}, 512)"
            return frame_embeds
    raise FileNotFoundError(f"No cache for '{video_id}' under {panda_dir}/(test|train)")


def resolve_result_csv_path(result_file):
    if os.path.isabs(result_file):
        return result_file
    return os.path.join("output", "evaluation_results", "rerank", result_file)


def write_result_csv(result_file, metrics):
    csv_path = resolve_result_csv_path(result_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['R@1', 'R@5', 'R@10', 'MedR', 'MeanR'])
        writer.writerow([
            metrics['R@1'],
            metrics['R@5'],
            metrics['R@10'],
            metrics['MedR'],
            metrics['MeanR'],
        ])
    print(f"wrote {csv_path}")


def encode_texts(model, tokenizer, texts, device, batch_size):
    """HF CLIP text encode, batched -> [Nt, 512] (no normalization here, matches test.py)."""
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start:start + batch_size], return_tensors='pt',
                              padding=True, truncation=True)
            batch = {k: v.to(device) for k, v in batch.items()}
            feats.append(model.clip.get_text_features(**batch).cpu())
    return torch.cat(feats)


def main():
    args = parse_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    model = build_model(args.eval_checkpoint, device)

    results = json.load(open(args.candidate_file))['results']
    texts = [r['query_text'] for r in results]
    text_features = encode_texts(model, tokenizer, texts, device, args.batch_size)

    cap = args.max_candidates
    def eff_cands(row):
        """Top-K stage-1 candidates (candidate lists are pre-sorted by stage-1 score)."""
        return row['candidates'][:cap] if cap and cap > 0 else row['candidates']

    # Per-video frame cache: pre-load all unique candidate/gt ids in parallel.
    # np.load per-file overhead (npz unzip) dominates and releases the GIL, so threads scale well.
    from concurrent.futures import ThreadPoolExecutor
    need = set()
    for r in results:
        cands = eff_cands(r)
        if r['ground_truth_video_id'] in cands:
            need.update(cands)
            need.add(r['ground_truth_video_id'])
    frame_by_id = {}
    with ThreadPoolExecutor(max_workers=args.load_workers) as ex:
        for vid, fr in ex.map(lambda v: (v, load_frame_cache(v, args.video_cache_dir, args.num_frames)),
                              need):
            frame_by_id[vid] = fr
    print(f"pre-loaded {len(frame_by_id)} unique frame caches ({args.load_workers} workers)")

    def get_frames(vid):
        return frame_by_id[vid]

    ranks = []  # 0-based rank of GT among candidates; n_cand if GT absent
    cand_counts = []
    with torch.no_grad():
        for i, row in enumerate(results):
            cands = eff_cands(row)
            n_cand = len(cands)
            cand_counts.append(n_cand)
            gt = row['ground_truth_video_id']
            if gt not in cands:
                ranks.append(n_cand)
                continue
            cand_frames = torch.from_numpy(
                np.stack([get_frames(c) for c in cands])).float().to(device)  # [K,4,512]
            text_i = text_features[i:i + 1].to(device)                        # [1,512]
            pooled = model.pool_frames(text_i, cand_frames).squeeze(1)        # [K,512]
            text_norm = text_i / text_i.norm(dim=-1, keepdim=True)
            pooled_norm = pooled / pooled.norm(dim=-1, keepdim=True)
            sims = (text_norm * pooled_norm).sum(dim=-1)                      # [K]
            gt_idx = cands.index(gt)
            ranks.append(int((sims > sims[gt_idx]).sum().item()))

    ranks = np.array(ranks)
    n = len(ranks)
    metrics = {
        "R@1": 100.0 * float(np.sum(ranks == 0)) / n,
        "R@5": 100.0 * float(np.sum(ranks < 5)) / n,
        "R@10": 100.0 * float(np.sum(ranks < 10)) / n,
        # MedR/MeanR over all queries; GT-absent contributes its candidate count (a miss for any K).
        "MedR": float(np.median(ranks) + 1),
        "MeanR": float(np.mean(ranks) + 1),
        # Effective candidate budget actually reranked (post-cap), for honest budget reporting.
        "avg_candidates_used": float(np.mean(cand_counts)),
    }

    if args.out_json:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        json.dump({"metadata": {"candidate_file": args.candidate_file,
                                "eval_checkpoint": args.eval_checkpoint,
                                "num_queries": n, "num_frames": args.num_frames,
                                "max_candidates": cap},
                   "metrics": metrics}, open(args.out_json, 'w'), indent=2)
        print(f"wrote {args.out_json} metrics={metrics}")
    if args.result_file:
        write_result_csv(args.result_file, metrics)


if __name__ == '__main__':
    main()
