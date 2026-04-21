"""ANN + Dense Retrieval baseline for text-to-video retrieval.

Stage-1 candidate selection using XPool-CLIP video embeddings + FAISS.
Designed to plug into the same Stage-2 X-Pool reranker as GRDR via
candidates/<ds>_ann_<idx>_<K>_candidates_t<setting>.json.

Key features:
- --per_query_timing: batch=1 encode and batch=1 search loop with warmup,
  emits per-query (T_text_encode, T_search) into the candidate JSON metadata
  so latency_report.py can stitch with Stage-2 numbers.
- ANN-baseline defaults for K via ANN_BASELINE_NUM_CANDIDATES[(dataset, setting)];
  the CLI --num_candidates overrides if set.
- Clip-suffix stripping (matches trainer/evaluator.py:711-743) when building
  pools, so the ANN pool size matches the GRDR pool size by construction.
"""
import argparse
import csv
import json
import os
import re
import time

import faiss
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer


# ANN baseline uses 100 candidates in both settings for all datasets.
ANN_BASELINE_NUM_CANDIDATES = {
    ('msrvtt', 1): 100, ('msrvtt', 2): 100,
    ('actnet', 1): 100, ('actnet', 2): 100,
    ('didemo', 1): 100, ('didemo', 2): 100,
    ('lsmdc',  1): 100, ('lsmdc',  2): 100,
}


def parse_args():
    parser = argparse.ArgumentParser(description="ANN Dense Retrieval Baseline")
    parser.add_argument('--dataset', type=str, default='msrvtt',
                        choices=['msrvtt', 'actnet', 'didemo', 'lsmdc'])
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                        help='1=test-only pool, 2=train+test combined pool')
    parser.add_argument('--index_type', type=str, default='all',
                        choices=['flat', 'hnsw', 'ivf', 'all'])
    parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/ckpt/msrvtt9k_model_best.pth')
    parser.add_argument('--cache_dir', type=str,
                        default='reranker/xpool/video_features_cache/Xpool')
    parser.add_argument('--output_dir', type=str, default='output/ann_baseline')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Encode/search batch (only used when --per_query_timing is OFF)')
    parser.add_argument('--hnsw_m', type=int, default=32)
    parser.add_argument('--hnsw_ef_search', type=int, default=128)
    parser.add_argument('--ivf_nlist', type=int, default=100)
    parser.add_argument('--ivf_nprobe', type=int, default=10)
    parser.add_argument('--num_candidates', type=int, default=None,
                        help='K for candidate output. Defaults to ANN_BASELINE_NUM_CANDIDATES[(ds,setting)] if unset.')
    parser.add_argument('--candidate_dir', type=str, default='candidates')
    parser.add_argument('--per_query_timing', action='store_true',
                        help='Run encode and FAISS search one query at a time '
                             'with warmup; record per-query timings in JSON metadata.')
    parser.add_argument('--num_warmup', type=int, default=10,
                        help='Warmup queries for --per_query_timing')
    return parser.parse_args()


# Dataset helpers
DATASET_NAME_MAP = {
    'msrvtt': 'MSRVTT', 'actnet': 'ACTNET',
    'didemo': 'DIDEMO', 'lsmdc': 'LSMDC',
}


def strip_clip_suffix(video_id):
    """Match trainer/evaluator.py:713-714: drop trailing _<=2-digit sample idx."""
    parts = video_id.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 2:
        return parts[0]
    return video_id


def dedup_by_base(video_ids):
    """First-wins dedup keyed on base id (clip-suffix stripped)."""
    seen = set()
    out = []
    for vid in video_ids:
        base = strip_clip_suffix(vid)
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


def load_test_queries(dataset):
    """Returns list of (video_id, caption_text). Video IDs are kept raw here;
    base-id stripping is applied later when matching against the pool."""
    if dataset == 'msrvtt':
        df = pd.read_csv('reranker/xpool/data/MSRVTT/MSRVTT_JSFUSION_test.csv')
        return [(row.video_id, row.sentence) for _, row in df.iterrows()]
    elif dataset == 'actnet':
        with open('reranker/xpool/data/ACTNET/actnet_ret_test.json') as f:
            data = json.load(f)
        return [(item['video'].replace('.mp4', ''),
                 re.sub(r'\s+', ' ', ' '.join(item['caption']).strip()))
                for item in data]
    elif dataset == 'didemo':
        with open('reranker/xpool/data/DIDEMO/didemo_ret_test.json') as f:
            data = json.load(f)
        return [(item['video'].replace('.mp4', ''),
                 re.sub(r'\s+', ' ', ' '.join(item['caption']).strip()))
                for item in data]
    elif dataset == 'lsmdc':
        pairs = []
        with open('reranker/xpool/data/LSMDC/LSMDC16_annos_test.csv') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    pairs.append((parts[0], parts[5]))
        return pairs


def load_train_video_ids(dataset):
    """Load training video IDs for Setting 2 (raw, before suffix stripping)."""
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
        clip_ids = []
        with open('reranker/xpool/data/LSMDC/LSMDC16_annos_training.csv') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    cid = parts[0]
                    if cid != '1012_Unbreakable_00.05.16.065-00.05.21.941':
                        clip_ids.append(cid)
        return clip_ids


# Feature loading
def load_video_embeddings(video_ids, cache_dir, dataset):
    """Mean-pool cached XPool frame embeddings to a single 512-d vector per video."""
    cache_path = os.path.join(cache_dir, DATASET_NAME_MAP[dataset])
    embs, valid = [], []
    missing = 0
    for vid in tqdm(video_ids, desc=f"Loading {dataset} video features"):
        npz_path = os.path.join(cache_path, f"{vid}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue
        data = np.load(npz_path)
        embs.append(data['frame_embeds'].mean(axis=0))
        valid.append(vid)
    if missing > 0:
        print(f"Warning: {missing}/{len(video_ids)} videos missing from cache")
    return np.array(embs, dtype=np.float32), valid


def encode_text_queries_batched(captions, model, tokenizer, device, batch_size=64):
    """Batch encode for the throughput path (non per_query_timing)."""
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


def encode_text_per_query(caption, model, tokenizer, device):
    """Single-query encode; returns (numpy_embed [1, D], elapsed_seconds)."""
    inputs = tokenizer([caption], return_tensors='pt', padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return emb.cpu().numpy().astype(np.float32), elapsed


# FAISS index builders
def normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def build_flat_index(embeddings):
    """Exact inner product on L2-normalized vectors (cosine similarity)."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(normalize(embeddings))
    return index


def build_hnsw_index(embeddings, M=32, ef_search=128):
    """HNSW approximate nearest neighbor (inner product)."""
    index = faiss.IndexHNSWFlat(embeddings.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(normalize(embeddings))
    return index


def build_ivf_index(embeddings, nlist=100, nprobe=10):
    """IVF-Flat approximate nearest neighbor (inner product)."""
    n, dim = embeddings.shape
    nlist = min(nlist, max(1, n // 4))
    nprobe = min(nprobe, nlist)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    normed = normalize(embeddings)
    index.train(normed)
    index.add(normed)
    index.nprobe = nprobe
    return index


# Evaluation
def compute_metrics(ranks):
    return {
        'R@1': 100.0 * float(np.mean(ranks < 1)),
        'R@5': 100.0 * float(np.mean(ranks < 5)),
        'R@10': 100.0 * float(np.mean(ranks < 10)),
        'MedR': float(np.median(ranks) + 1),
        'MeanR': float(np.mean(ranks) + 1),
    }


def search_batched(query_embs, index, gt_indices, k):
    """Throughput path: search all queries in one call."""
    qn = normalize(query_embs)
    if hasattr(faiss, 'omp_set_num_threads'):
        pass  # leave thread count to global default
    t0 = time.perf_counter()
    distances, retrieved = index.search(qn, k)
    search_time = time.perf_counter() - t0
    n = len(gt_indices)
    ranks = np.full(n, k + 1, dtype=np.float64)
    for i in range(n):
        m = np.where(retrieved[i] == gt_indices[i])[0]
        if len(m) > 0:
            ranks[i] = m[0]
    metrics = compute_metrics(ranks)
    metrics['search_time_total_s'] = search_time
    metrics['search_time_per_query_ms'] = 1000.0 * search_time / max(n, 1)
    metrics['queries_per_sec'] = n / search_time if search_time > 0 else 0
    return metrics, retrieved, distances, None  # last slot reserved for per-query timings


def search_per_query(captions, model, tokenizer, device, index, gt_indices,
                     k, num_warmup):
    """Per-query path: batch=1 encode + batch=1 search, with warmup. Returns
    metrics, retrieved [n,k], distances [n,k], and per-query timings list."""
    n = len(captions)
    retrieved = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), -np.inf, dtype=np.float32)
    timings = []  # list of dicts: {encode_s, search_s}

    # Warmup using first few queries to settle CUDA kernels and OS cache.
    print(f"  Warmup: {num_warmup} queries (excluded from timing)")
    for i in range(min(num_warmup, n)):
        emb, _ = encode_text_per_query(captions[i], model, tokenizer, device)
        _ = index.search(normalize(emb), k)

    print(f"  Per-query timing: {n} queries (batch=1 encode + batch=1 search)")
    for i in tqdm(range(n), desc="Per-query"):
        emb, t_enc = encode_text_per_query(captions[i], model, tokenizer, device)
        emb_norm = normalize(emb)
        t0 = time.perf_counter()
        d, r = index.search(emb_norm, k)
        t_search = time.perf_counter() - t0
        retrieved[i] = r[0]
        distances[i] = d[0]
        timings.append({'encode_s': t_enc, 'search_s': t_search})

    ranks = np.full(n, k + 1, dtype=np.float64)
    for i in range(n):
        m = np.where(retrieved[i] == gt_indices[i])[0]
        if len(m) > 0:
            ranks[i] = m[0]
    metrics = compute_metrics(ranks)
    enc_arr = np.array([t['encode_s'] for t in timings])
    sea_arr = np.array([t['search_s'] for t in timings])
    metrics['encode_time_per_query_ms_mean'] = float(1000.0 * enc_arr.mean())
    metrics['encode_time_per_query_ms_std'] = float(1000.0 * enc_arr.std())
    metrics['search_time_per_query_ms_mean'] = float(1000.0 * sea_arr.mean())
    metrics['search_time_per_query_ms_std'] = float(1000.0 * sea_arr.std())
    metrics['search_time_total_s'] = float(sea_arr.sum())
    metrics['queries_per_sec'] = float(n / max(sea_arr.sum(), 1e-9))
    return metrics, retrieved, distances, timings


def save_candidate_json(retrieved, distances, valid_captions, valid_gt_vids,
                        pool_ids, num_candidates, metrics, args, idx_type,
                        out_path, per_query_timings=None):
    """Save candidate JSON in the same shape used by reranker/xpool/test.py."""
    results = []
    for i in range(len(valid_captions)):
        cands = [pool_ids[j] for j in retrieved[i, :num_candidates] if j >= 0]
        scores = [float(distances[i, j])
                  for j in range(min(num_candidates, len(cands)))]
        result_entry = {
            'query_text': valid_captions[i],
            'ground_truth_video_id': valid_gt_vids[i],
            'candidates': cands,
            'scores': scores,
            'num_candidates': len(cands),
        }
        if per_query_timings is not None:
            t = per_query_timings[i]
            result_entry['stage1_timing_ms'] = {
                'text_encode': 1000.0 * t['encode_s'],
                'search': 1000.0 * t['search_s'],
            }
        results.append(result_entry)

    gt_in = sum(1 for r in results
                if r['ground_truth_video_id'] in r['candidates'])
    recall_at_k = gt_in / len(results) if results else 0
    metadata = {
        'dataset': args.dataset,
        'setting': args.setting,
        'index_type': idx_type,
        'num_candidates': num_candidates,
        'pool_size': len(pool_ids),
        'method': 'ann_dense_retrieval',
        'per_query_timing': per_query_timings is not None,
    }
    if per_query_timings is not None:
        metadata['stage1_latency_ms'] = {
            'text_encode_mean': metrics.get('encode_time_per_query_ms_mean'),
            'text_encode_std': metrics.get('encode_time_per_query_ms_std'),
            'search_mean': metrics.get('search_time_per_query_ms_mean'),
            'search_std': metrics.get('search_time_per_query_ms_std'),
        }
    output = {
        'metadata': metadata,
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
    print(f"  Recall@{num_candidates}: {recall_at_k:.4f} ({gt_in}/{len(results)})")


# Main
def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.num_candidates is None:
        args.num_candidates = ANN_BASELINE_NUM_CANDIDATES[(args.dataset, args.setting)]
        print(f"Using ANN baseline K = {args.num_candidates} for ({args.dataset}, t{args.setting})")

    print(f"\n{'='*70}")
    print(f"ANN Dense Retrieval Baseline")
    print(f"Dataset: {args.dataset}, Setting: {args.setting}, Index: {args.index_type}, K={args.num_candidates}")
    print(f"Per-query timing: {args.per_query_timing}")
    print(f"{'='*70}\n")

    # 1. Load test queries (raw IDs).
    test_pairs = load_test_queries(args.dataset)
    raw_test_ids = [vid for vid, _ in test_pairs]
    test_captions = [cap for _, cap in test_pairs]
    # Strip clip-suffix and dedupe to get the test pool base IDs.
    unique_test_base = dedup_by_base(raw_test_ids)
    print(f"Test queries: {len(test_pairs)}, Unique base test videos: {len(unique_test_base)}")

    # 2. Build pool (test first, then train extras for setting 2).
    if args.setting == 2:
        train_video_ids = load_train_video_ids(args.dataset)
        train_base_unique = dedup_by_base(train_video_ids)
        test_base_set = set(unique_test_base)
        pool_video_ids = list(unique_test_base)
        for b in train_base_unique:
            if b not in test_base_set:
                pool_video_ids.append(b)
        print(f"Setting 2: {len(unique_test_base)} test + "
              f"{len(pool_video_ids) - len(unique_test_base)} train = "
              f"{len(pool_video_ids)} total (clip-suffix stripped)")
    else:
        pool_video_ids = unique_test_base
        print(f"Setting 1: {len(pool_video_ids)} test base videos in pool")

    # 3. Load video embeddings (cache files keyed by base id).
    video_embs, valid_pool_ids = load_video_embeddings(
        pool_video_ids, args.cache_dir, args.dataset)
    print(f"Loaded {len(valid_pool_ids)} video embeddings, dim={video_embs.shape[1]}")

    vid_to_pool_idx = {vid: i for i, vid in enumerate(valid_pool_ids)}

    # 4. Build GT indices using base-stripped query video ids.
    gt_indices, valid_query_mask, valid_gt_vids = [], [], []
    for vid, _ in test_pairs:
        base = strip_clip_suffix(vid)
        if base in vid_to_pool_idx:
            gt_indices.append(vid_to_pool_idx[base])
            valid_query_mask.append(True)
            valid_gt_vids.append(base)
        else:
            valid_query_mask.append(False)
    gt_indices = np.array(gt_indices)
    print(f"Valid queries with GT in pool: {len(gt_indices)}/{len(test_pairs)}")

    # 5. Load CLIP weights from XPool checkpoint.
    print(f"\nLoading X-Pool checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    clip_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('clip.')}
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.load_state_dict(clip_state_dict, strict=False)
    clip_model = clip_model.to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    valid_captions = [cap for cap, ok in zip(test_captions, valid_query_mask) if ok]

    # Throughput-mode pre-encoding (only when not in per-query timing mode).
    if not args.per_query_timing:
        query_embs = encode_text_queries_batched(
            valid_captions, clip_model, tokenizer, device, args.batch_size)
        print(f"Encoded {len(valid_captions)} text queries, dim={query_embs.shape[1]}")

    # 6. Build indices and evaluate.
    search_k = max(100, args.num_candidates)
    index_types = ['flat', 'hnsw', 'ivf'] if args.index_type == 'all' else [args.index_type]
    all_results = {}

    for idx_type in index_types:
        print(f"\n--- {idx_type.upper()} Index ---")
        t0 = time.perf_counter()
        if idx_type == 'flat':
            index = build_flat_index(video_embs)
        elif idx_type == 'hnsw':
            index = build_hnsw_index(video_embs, M=args.hnsw_m,
                                     ef_search=args.hnsw_ef_search)
        elif idx_type == 'ivf':
            index = build_ivf_index(video_embs, nlist=args.ivf_nlist,
                                    nprobe=args.ivf_nprobe)
        build_time = time.perf_counter() - t0
        print(f"Index built in {build_time:.2f}s ({index.ntotal} vectors)")

        if args.per_query_timing:
            metrics, retrieved, distances, per_query_timings = search_per_query(
                valid_captions, clip_model, tokenizer, device, index,
                gt_indices, search_k, args.num_warmup)
        else:
            metrics, retrieved, distances, per_query_timings = search_batched(
                query_embs, index, gt_indices, search_k)

        metrics['build_time_s'] = build_time
        all_results[idx_type] = metrics

        print(f"R@1: {metrics['R@1']:.2f}  R@5: {metrics['R@5']:.2f}  "
              f"R@10: {metrics['R@10']:.2f}  MedR: {metrics['MedR']:.1f}  "
              f"MeanR: {metrics['MeanR']:.1f}")
        if args.per_query_timing:
            print(f"Encode/q: {metrics['encode_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['encode_time_per_query_ms_std']:.2f} ms; "
                  f"Search/q: {metrics['search_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['search_time_per_query_ms_std']:.2f} ms")
        else:
            print(f"Search: {metrics['search_time_total_s']:.4f}s "
                  f"({metrics['queries_per_sec']:.0f} q/s), "
                  f"Build: {build_time:.2f}s")

        cand_filename = (f"{args.dataset}_ann_{idx_type}"
                         f"_{args.num_candidates}_candidates"
                         f"_t{args.setting}.json")
        cand_path = os.path.join(args.candidate_dir, cand_filename)
        save_candidate_json(retrieved, distances, valid_captions, valid_gt_vids,
                            valid_pool_ids, args.num_candidates, metrics, args,
                            idx_type, cand_path,
                            per_query_timings=per_query_timings)

    # 7. Save run summary.
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = '_pqt' if args.per_query_timing else ''
    result_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_setting{args.setting}{suffix}_results.json")
    with open(result_file, 'w') as f:
        json.dump({
            'dataset': args.dataset,
            'setting': args.setting,
            'num_candidates': args.num_candidates,
            'num_queries': len(valid_captions),
            'pool_size': len(valid_pool_ids),
            'per_query_timing': args.per_query_timing,
            'results': {k: {mk: float(mv) for mk, mv in v.items()}
                        for k, v in all_results.items()},
        }, f, indent=2)
    print(f"\nResults saved to: {result_file}")

    csv_file = os.path.join(
        args.output_dir,
        f"{args.dataset}_setting{args.setting}{suffix}_results.csv")
    with open(csv_file, 'w', newline='') as f:
        w = csv.writer(f)
        if args.per_query_timing:
            w.writerow(['index_type', 'R@1', 'R@5', 'R@10', 'MedR', 'MeanR',
                        'enc_ms_mean', 'enc_ms_std',
                        'search_ms_mean', 'search_ms_std', 'build_s'])
            for idx_type, m in all_results.items():
                w.writerow([idx_type, f"{m['R@1']:.2f}", f"{m['R@5']:.2f}",
                            f"{m['R@10']:.2f}", f"{m['MedR']:.1f}",
                            f"{m['MeanR']:.1f}",
                            f"{m['encode_time_per_query_ms_mean']:.3f}",
                            f"{m['encode_time_per_query_ms_std']:.3f}",
                            f"{m['search_time_per_query_ms_mean']:.3f}",
                            f"{m['search_time_per_query_ms_std']:.3f}",
                            f"{m['build_time_s']:.2f}"])
        else:
            w.writerow(['index_type', 'R@1', 'R@5', 'R@10', 'MedR', 'MeanR',
                        'search_time_s', 'build_s'])
            for idx_type, m in all_results.items():
                w.writerow([idx_type, f"{m['R@1']:.2f}", f"{m['R@5']:.2f}",
                            f"{m['R@10']:.2f}", f"{m['MedR']:.1f}",
                            f"{m['MeanR']:.1f}",
                            f"{m['search_time_total_s']:.4f}",
                            f"{m['build_time_s']:.2f}"])
    print(f"CSV saved to: {csv_file}")


if __name__ == '__main__':
    main()
