"""ANN + Dense Retrieval baseline for text-to-video retrieval.

Uses X-Pool (CLIP-based) embeddings with FAISS ANN search.
Video features: mean-pooled frame embeddings from Xpool cache.
Text features: encoded via CLIP text encoder from X-Pool checkpoint.
"""
import argparse
import csv
import json
import os
import re
import sys
import time

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="ANN Dense Retrieval Baseline")
    parser.add_argument('--dataset', type=str, default='msrvtt',
                        choices=['msrvtt', 'actnet', 'didemo', 'lsmdc'])
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                        help='1=test-only pool, 2=train+test combined pool')
    parser.add_argument('--index_type', type=str, default='all',
                        choices=['flat', 'hnsw', 'ivf', 'all'],
                        help='FAISS index type')
    parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/ckpt/msrvtt9k_model_best.pth')
    parser.add_argument('--cache_dir', type=str,
                        default='reranker/xpool/video_features_cache/Xpool')
    parser.add_argument('--output_dir', type=str, default='output/ann_baseline')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    # HNSW params
    parser.add_argument('--hnsw_m', type=int, default=32, help='HNSW M parameter')
    parser.add_argument('--hnsw_ef_search', type=int, default=128,
                        help='HNSW efSearch parameter')
    # IVF params
    parser.add_argument('--ivf_nlist', type=int, default=100, help='IVF nlist')
    parser.add_argument('--ivf_nprobe', type=int, default=10, help='IVF nprobe')
    # Candidate output for two-stage pipeline
    parser.add_argument('--num_candidates', type=int, default=50,
                        help='Number of candidates to retrieve per query for reranking')
    parser.add_argument('--candidate_dir', type=str, default='candidates',
                        help='Directory to save candidate JSON files')
    return parser.parse_args()


# ── Dataset helpers ──────────────────────────────────────────────────────────

DATASET_NAME_MAP = {
    'msrvtt': 'MSRVTT', 'actnet': 'ACTNET',
    'didemo': 'DIDEMO', 'lsmdc': 'LSMDC',
}


def load_test_queries(dataset):
    """Load test queries: returns list of (video_id, caption_text)."""
    ds = DATASET_NAME_MAP[dataset]

    if dataset == 'msrvtt':
        test_csv = 'reranker/xpool/data/MSRVTT/MSRVTT_JSFUSION_test.csv'
        df = pd.read_csv(test_csv)
        return [(row.video_id, row.sentence) for _, row in df.iterrows()]

    elif dataset == 'actnet':
        with open('reranker/xpool/data/ACTNET/actnet_ret_test.json') as f:
            data = json.load(f)
        pairs = []
        for item in data:
            vid = item['video'].replace('.mp4', '')
            caption = re.sub(r'\s+', ' ', ' '.join(item['caption']).strip())
            pairs.append((vid, caption))
        return pairs

    elif dataset == 'didemo':
        with open('reranker/xpool/data/DIDEMO/didemo_ret_test.json') as f:
            data = json.load(f)
        pairs = []
        for item in data:
            vid = item['video'].replace('.mp4', '')
            caption = re.sub(r'\s+', ' ', ' '.join(item['caption']).strip())
            pairs.append((vid, caption))
        return pairs

    elif dataset == 'lsmdc':
        test_file = 'reranker/xpool/data/LSMDC/LSMDC16_annos_test.csv'
        pairs = []
        with open(test_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    clip_id = parts[0]
                    caption = parts[5]
                    pairs.append((clip_id, caption))
        return pairs


def load_train_video_ids(dataset):
    """Load training video IDs for Setting 2."""
    if dataset == 'msrvtt':
        df = pd.read_csv('reranker/xpool/data/MSRVTT/MSRVTT_train.9k.csv')
        return df['video_id'].unique().tolist()

    elif dataset == 'actnet':
        with open('reranker/xpool/data/ACTNET/actnet_ret_train.json') as f:
            data = json.load(f)
        return [item['video'].replace('.mp4', '') for item in data]

    elif dataset == 'didemo':
        with open('reranker/xpool/data/DIDEMO/didemo_ret_train.json') as f:
            data = json.load(f)
        return [item['video'].replace('.mp4', '') for item in data]

    elif dataset == 'lsmdc':
        train_file = 'reranker/xpool/data/LSMDC/LSMDC16_annos_training.csv'
        clip_ids = []
        with open(train_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    clip_id = parts[0]
                    if clip_id != '1012_Unbreakable_00.05.16.065-00.05.21.941':
                        clip_ids.append(clip_id)
        return list(dict.fromkeys(clip_ids))  # deduplicate preserving order


# ── Feature loading ──────────────────────────────────────────────────────────

def load_video_embeddings(video_ids, cache_dir, dataset):
    """Load and mean-pool cached Xpool frame embeddings."""
    ds = DATASET_NAME_MAP[dataset]
    cache_path = os.path.join(cache_dir, ds)

    embeddings = []
    valid_ids = []
    missing = 0
    for vid in tqdm(video_ids, desc=f"Loading {ds} video features"):
        npz_path = os.path.join(cache_path, f"{vid}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue
        data = np.load(npz_path)
        frame_embs = data['frame_embeds']  # (num_frames, 512)
        pooled = frame_embs.mean(axis=0)   # (512,)
        embeddings.append(pooled)
        valid_ids.append(vid)

    if missing > 0:
        print(f"Warning: {missing}/{len(video_ids)} videos missing from cache")

    return np.array(embeddings, dtype=np.float32), valid_ids


def encode_text_queries(captions, model, tokenizer, device, batch_size=64):
    """Encode text queries using CLIP text encoder."""
    model.eval()
    all_embeds = []
    for i in tqdm(range(0, len(captions), batch_size), desc="Encoding text queries"):
        batch = captions[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            text_embeds = model.get_text_features(**inputs)
        all_embeds.append(text_embeds.cpu().numpy())
    return np.concatenate(all_embeds, axis=0).astype(np.float32)


# ── FAISS index builders ─────────────────────────────────────────────────────

def normalize(x):
    """L2-normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def build_flat_index(embeddings):
    """Exact inner product search (cosine sim on normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(normalize(embeddings))
    return index


def build_hnsw_index(embeddings, M=32, ef_search=128):
    """HNSW approximate nearest neighbor index."""
    dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(normalize(embeddings))
    return index


def build_ivf_index(embeddings, nlist=100, nprobe=10):
    """IVF-Flat approximate nearest neighbor index."""
    dim = embeddings.shape[1]
    n = embeddings.shape[0]
    # Adjust nlist if corpus is small
    nlist = min(nlist, max(1, n // 4))
    nprobe = min(nprobe, nlist)

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    normed = normalize(embeddings)
    index.train(normed)
    index.add(normed)
    index.nprobe = nprobe
    return index


# ── Evaluation ───────────────────────────────────────────────────────────────

def compute_metrics(ranks):
    """Compute retrieval metrics from rank array."""
    r1 = 100.0 * np.mean(ranks < 1)
    r5 = 100.0 * np.mean(ranks < 5)
    r10 = 100.0 * np.mean(ranks < 10)
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1
    return {'R@1': r1, 'R@5': r5, 'R@10': r10, 'MedR': medr, 'MeanR': meanr}


def search_and_evaluate(query_embs, index, gt_indices, k=100):
    """Search index and compute metrics. Returns metrics, retrieved indices, and distances."""
    query_normed = normalize(query_embs)
    t0 = time.time()
    distances, retrieved = index.search(query_normed, k)
    search_time = time.time() - t0

    # Compute rank of ground truth for each query
    num_queries = len(gt_indices)
    ranks = np.full(num_queries, k + 1, dtype=np.float64)
    for i in range(num_queries):
        gt = gt_indices[i]
        matches = np.where(retrieved[i] == gt)[0]
        if len(matches) > 0:
            ranks[i] = matches[0]

    metrics = compute_metrics(ranks)
    metrics['search_time'] = search_time
    metrics['queries_per_sec'] = num_queries / search_time if search_time > 0 else 0
    return metrics, retrieved, distances


def save_candidate_json(retrieved, distances, valid_captions, valid_gt_vids,
                        pool_ids, num_candidates, metrics, args, idx_type, out_path):
    """Save candidate JSON in X-Pool reranking format."""
    results = []
    for i in range(len(valid_captions)):
        cands = [pool_ids[j] for j in retrieved[i, :num_candidates] if j >= 0]
        scores = [float(distances[i, j]) for j in range(min(num_candidates, len(cands)))]
        results.append({
            'query_text': valid_captions[i],
            'ground_truth_video_id': valid_gt_vids[i],
            'candidates': cands,
            'scores': scores,
            'num_candidates': len(cands),
        })

    # Compute recall@num_candidates
    gt_in_candidates = sum(
        1 for r in results if r['ground_truth_video_id'] in r['candidates']
    )
    recall_at_k = gt_in_candidates / len(results) if results else 0

    output = {
        'metadata': {
            'dataset': args.dataset,
            'setting': args.setting,
            'index_type': idx_type,
            'num_candidates': num_candidates,
            'pool_size': len(pool_ids),
            'method': 'ann_dense_retrieval',
        },
        'metrics': {
            f'Recall@{num_candidates}': recall_at_k,
            'Recall@1': metrics['R@1'] / 100,
            'Recall@5': metrics['R@5'] / 100,
            'Recall@10': metrics['R@10'] / 100,
            'total_queries': len(results),
        },
        'results': results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Candidate JSON ({num_candidates} per query) saved to: {out_path}")
    print(f"  Recall@{num_candidates}: {recall_at_k:.4f} ({gt_in_candidates}/{len(results)})")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*70}")
    print(f"ANN Dense Retrieval Baseline")
    print(f"Dataset: {args.dataset}, Setting: {args.setting}, Index: {args.index_type}")
    print(f"{'='*70}\n")

    # 1. Load test queries
    test_pairs = load_test_queries(args.dataset)
    test_video_ids = [vid for vid, _ in test_pairs]
    test_captions = [cap for _, cap in test_pairs]
    # Deduplicate test video IDs while preserving order
    seen = set()
    unique_test_vids = []
    for vid in test_video_ids:
        if vid not in seen:
            seen.add(vid)
            unique_test_vids.append(vid)

    print(f"Test queries: {len(test_pairs)}, Unique test videos: {len(unique_test_vids)}")

    # 2. Build video pool
    if args.setting == 2:
        train_video_ids = load_train_video_ids(args.dataset)
        # Deduplicate train video IDs
        train_vids_dedup = list(dict.fromkeys(train_video_ids))
        # Combined pool: test first, then train (no duplicates)
        pool_video_ids = unique_test_vids.copy()
        test_vid_set = set(unique_test_vids)
        for vid in train_vids_dedup:
            if vid not in test_vid_set:
                pool_video_ids.append(vid)
        print(f"Setting 2: {len(unique_test_vids)} test + {len(pool_video_ids) - len(unique_test_vids)} train = {len(pool_video_ids)} total")
    else:
        pool_video_ids = unique_test_vids
        print(f"Setting 1: {len(pool_video_ids)} test videos in pool")

    # 3. Load video embeddings
    video_embs, valid_pool_ids = load_video_embeddings(
        pool_video_ids, args.cache_dir, args.dataset)
    print(f"Loaded {len(valid_pool_ids)} video embeddings, dim={video_embs.shape[1]}")

    # Build video_id -> pool_index mapping
    vid_to_pool_idx = {vid: i for i, vid in enumerate(valid_pool_ids)}

    # 4. Build ground truth indices and track valid GT video IDs
    gt_indices = []
    valid_query_mask = []
    valid_gt_vids = []
    for vid, cap in test_pairs:
        if vid in vid_to_pool_idx:
            gt_indices.append(vid_to_pool_idx[vid])
            valid_query_mask.append(True)
            valid_gt_vids.append(vid)
        else:
            valid_query_mask.append(False)
    gt_indices = np.array(gt_indices)
    print(f"Valid queries with GT in pool: {len(gt_indices)}/{len(test_pairs)}")

    # 5. Load CLIP model and encode text queries
    print(f"\nLoading X-Pool checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']

    # Extract CLIP weights from X-Pool checkpoint
    clip_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('clip.'):
            clip_state_dict[k[5:]] = v  # remove 'clip.' prefix

    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.load_state_dict(clip_state_dict, strict=False)
    clip_model = clip_model.to(device)
    clip_model.eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Only encode valid queries
    valid_captions = [cap for cap, valid in zip(test_captions, valid_query_mask) if valid]
    query_embs = encode_text_queries(valid_captions, clip_model, tokenizer,
                                     device, args.batch_size)
    print(f"Encoded {len(valid_captions)} text queries, dim={query_embs.shape[1]}")

    # Free GPU memory
    del clip_model
    torch.cuda.empty_cache()

    # 6. Build indices, evaluate, and save candidate JSONs
    search_k = max(100, args.num_candidates)
    index_types = ['flat', 'hnsw', 'ivf'] if args.index_type == 'all' else [args.index_type]
    all_results = {}

    for idx_type in index_types:
        print(f"\n--- {idx_type.upper()} Index ---")
        t0 = time.time()
        if idx_type == 'flat':
            index = build_flat_index(video_embs)
        elif idx_type == 'hnsw':
            index = build_hnsw_index(video_embs, M=args.hnsw_m,
                                     ef_search=args.hnsw_ef_search)
        elif idx_type == 'ivf':
            index = build_ivf_index(video_embs, nlist=args.ivf_nlist,
                                    nprobe=args.ivf_nprobe)
        build_time = time.time() - t0
        print(f"Index built in {build_time:.2f}s ({index.ntotal} vectors)")

        metrics, retrieved, distances = search_and_evaluate(
            query_embs, index, gt_indices, k=search_k)
        metrics['build_time'] = build_time
        all_results[idx_type] = metrics

        print(f"R@1: {metrics['R@1']:.2f}  R@5: {metrics['R@5']:.2f}  "
              f"R@10: {metrics['R@10']:.2f}  MedR: {metrics['MedR']:.1f}  "
              f"MeanR: {metrics['MeanR']:.1f}")
        print(f"Search: {metrics['search_time']:.4f}s "
              f"({metrics['queries_per_sec']:.0f} q/s), "
              f"Build: {build_time:.2f}s")

        # Save candidate JSON for X-Pool reranking
        cand_filename = (f"{args.dataset}_ann_{idx_type}"
                         f"_{args.num_candidates}_candidates"
                         f"_t{args.setting}.json")
        cand_path = os.path.join(args.candidate_dir, cand_filename)
        save_candidate_json(retrieved, distances, valid_captions, valid_gt_vids,
                            valid_pool_ids, args.num_candidates, metrics, args,
                            idx_type, cand_path)

    # 7. Save results
    os.makedirs(args.output_dir, exist_ok=True)
    result_file = os.path.join(
        args.output_dir, f"{args.dataset}_setting{args.setting}_results.json")
    output = {
        'dataset': args.dataset,
        'setting': args.setting,
        'num_queries': len(valid_captions),
        'pool_size': len(valid_pool_ids),
        'results': {k: {mk: float(mv) for mk, mv in v.items()}
                    for k, v in all_results.items()},
    }
    with open(result_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    # Also save CSV summary
    csv_file = os.path.join(
        args.output_dir, f"{args.dataset}_setting{args.setting}_results.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index_type', 'R@1', 'R@5', 'R@10', 'MedR', 'MeanR',
                         'search_time', 'build_time'])
        for idx_type, m in all_results.items():
            writer.writerow([idx_type, f"{m['R@1']:.2f}", f"{m['R@5']:.2f}",
                           f"{m['R@10']:.2f}", f"{m['MedR']:.1f}",
                           f"{m['MeanR']:.1f}", f"{m['search_time']:.4f}",
                           f"{m['build_time']:.2f}"])
    print(f"CSV saved to: {csv_file}")


if __name__ == '__main__':
    main()
