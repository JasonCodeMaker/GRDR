"""ANN + Dense Retrieval baseline for text-to-video retrieval.

Stage-1 candidate selection uses XPool-CLIP video embeddings plus offline ANN
structures. The intended pipeline is:
1. Offline index construction: load pooled video embeddings from cache, build
   the ANN structure, and persist only structure + mapping to disk.
2. Online query handling: encode one query, use the saved structure to decide
   which videos to inspect, load those videos' frame_embeds from disk, and
   export candidates for Stage 2.

Designed to plug into the same Stage-2 X-Pool reranker as GRDR via
candidates/<ds>_ann_<idx>_<K>_candidates_t<setting>.json.
"""
import argparse
import csv
import heapq
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
SUPPORTED_INDEX_TYPES = ('hnsw', 'ivf')


def parse_args():
    parser = argparse.ArgumentParser(description="ANN Dense Retrieval Baseline")
    parser.add_argument('--dataset', type=str, default='msrvtt',
                        choices=['msrvtt', 'actnet', 'didemo', 'lsmdc'])
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2],
                        help='1=test-only pool, 2=train+test combined pool')
    parser.add_argument('--index_type', type=str, default='all',
                        choices=[*SUPPORTED_INDEX_TYPES, 'all'])
    parser.add_argument('--checkpoint', type=str,
                        default='reranker/xpool/ckpt/msrvtt9k_model_best.pth')
    parser.add_argument('--cache_dir', type=str,
                        default='reranker/xpool/video_features_cache/Xpool')
    parser.add_argument('--output_dir', type=str, default='output/ann_baseline')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='Directory to store offline ANN index artifacts. '
                             'Defaults to <output_dir>/indexes.')
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
    parser.add_argument('--query_manifest', type=str, default=None,
                        help='Optional TSV with query_text and ground_truth_video_id columns. '
                             'When set, this defines the exact exported query row set.')
    parser.add_argument('--per_query_timing', action='store_true',
                        help='Run online evaluation one query at a time; each '
                             'query reloads video features from disk and records '
                             'per-query timings in JSON metadata.')
    parser.add_argument('--num_warmup', type=int, default=0,
                        help='Warmup queries for --per_query_timing')
    return parser.parse_args()


# Dataset helpers
DATASET_NAME_MAP = {
    'msrvtt': 'MSRVTT', 'actnet': 'ACTNET',
    'didemo': 'DIDEMO', 'lsmdc': 'LSMDC',
}


def resolve_dataset_cache_dir(cache_dir, dataset):
    dataset_name = DATASET_NAME_MAP[dataset]
    normalized_cache_dir = os.path.normpath(cache_dir)
    if os.path.basename(normalized_cache_dir).upper() == dataset_name.upper():
        return normalized_cache_dir
    return os.path.join(normalized_cache_dir, dataset_name)


def normalize_cache_video_id(video_id):
    if video_id.endswith('.mp4'):
        return video_id[:-4]
    if video_id.endswith('.avi'):
        return video_id[:-4]
    return video_id


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


def load_query_manifest(path):
    """Load package-owned query rows as (video_id, caption_text) pairs."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f, delimiter='\t'))
    missing = [col for col in ('ground_truth_video_id', 'query_text') if rows and col not in rows[0]]
    if missing:
        raise ValueError(f"query manifest {path} missing columns: {missing}")
    return [(row['ground_truth_video_id'], row['query_text']) for row in rows]


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
def pooled_video_vector(frame_embeds):
    pooled = frame_embeds.mean(axis=0).astype(np.float32, copy=False)
    norm = np.linalg.norm(pooled)
    if norm > 1e-8:
        pooled = pooled / norm
    return pooled


def load_video_embeddings(video_ids, cache_dir, dataset):
    """Mean-pool cached XPool frame embeddings to a single 512-d vector per video."""
    cache_path = resolve_dataset_cache_dir(cache_dir, dataset)
    embs, valid = [], []
    missing = 0
    for vid in tqdm(video_ids, desc=f"Loading {dataset} video features"):
        npz_path = os.path.join(cache_path, f"{normalize_cache_video_id(vid)}.npz")
        if not os.path.exists(npz_path):
            missing += 1
            continue
        with np.load(npz_path) as data:
            embs.append(pooled_video_vector(data['frame_embeds']))
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


def build_index(embeddings, idx_type, args):
    if idx_type == 'hnsw':
        return build_hnsw_index(embeddings, M=args.hnsw_m, ef_search=args.hnsw_ef_search)
    if idx_type == 'ivf':
        return build_ivf_index(embeddings, nlist=args.ivf_nlist, nprobe=args.ivf_nprobe)
    raise ValueError(f"Unsupported ANN index type: {idx_type}")


def index_path_for(index_dir, idx_type, dataset, setting):
    return os.path.join(index_dir, f"{dataset}_{idx_type}_setting{setting}.npz")


def index_meta_path_for(index_dir, idx_type, dataset, setting):
    return os.path.join(index_dir, f"{dataset}_{idx_type}_setting{setting}.meta.json")


def index_id_map_path_for(index_dir, idx_type, dataset, setting):
    return os.path.join(index_dir, f"{dataset}_{idx_type}_setting{setting}.id_map.json")


def extract_hnsw_structure(index):
    hnsw = index.hnsw
    max_level = int(hnsw.max_level)
    return {
        'entry_point': np.array([int(hnsw.entry_point)], dtype=np.int64),
        'max_level': np.array([max_level], dtype=np.int32),
        'levels': faiss.vector_to_array(hnsw.levels).astype(np.int32),
        'offsets': faiss.vector_to_array(hnsw.offsets).astype(np.int64),
        'neighbors': faiss.vector_to_array(hnsw.neighbors).astype(np.int32),
        'cum_nneighbor_per_level': np.array(
            [hnsw.cum_nb_neighbors(level) for level in range(max_level + 2)],
            dtype=np.int32,
        ),
    }


def extract_ivf_structure(index):
    centroids = np.stack(
        [index.quantizer.reconstruct(i) for i in range(index.nlist)],
        axis=0,
    ).astype(np.float32)
    list_offsets = [0]
    list_row_ids = []
    for list_no in range(index.nlist):
        list_size = index.invlists.list_size(list_no)
        if list_size > 0:
            ids_ptr = index.invlists.get_ids(list_no)
            try:
                ids = faiss.rev_swig_ptr(ids_ptr, list_size).astype(np.int64).copy()
            finally:
                index.invlists.release_ids(list_no, ids_ptr)
            list_row_ids.append(ids)
            list_offsets.append(list_offsets[-1] + list_size)
        else:
            list_offsets.append(list_offsets[-1])
    if list_row_ids:
        list_row_ids = np.concatenate(list_row_ids, axis=0)
    else:
        list_row_ids = np.empty((0,), dtype=np.int64)
    return {
        'centroids': centroids,
        'list_offsets': np.array(list_offsets, dtype=np.int64),
        'list_row_ids': list_row_ids,
    }


def extract_index_structure(index, idx_type):
    if idx_type == 'hnsw':
        return extract_hnsw_structure(index)
    if idx_type == 'ivf':
        return extract_ivf_structure(index)
    raise ValueError(f"Unsupported ANN index type: {idx_type}")


def save_index_artifacts(index, index_path, meta_path, id_map_path, idx_type, args, pool_ids,
                         video_embedding_load_time_s, build_time_s):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    np.savez_compressed(index_path, **extract_index_structure(index, idx_type))
    with open(id_map_path, 'w') as f:
        json.dump(pool_ids, f, indent=2)

    pool_size = len(pool_ids)

    metadata = {
        'dataset': args.dataset,
        'setting': args.setting,
        'index_type': idx_type,
        'pool_size': pool_size,
        'metric': 'inner_product',
        'normalized': True,
        'video_embedding_load_time_s': float(video_embedding_load_time_s),
        'build_time_s': float(build_time_s),
        'id_map_path': id_map_path,
    }
    if idx_type == 'hnsw':
        metadata.update({
            'hnsw_m': int(args.hnsw_m),
            'hnsw_ef_search': int(args.hnsw_ef_search),
        })
    elif idx_type == 'ivf':
        metadata.update({
            'ivf_nlist': int(min(args.ivf_nlist, max(1, pool_size // 4))),
            'ivf_nprobe': int(min(args.ivf_nprobe, min(args.ivf_nlist, max(1, pool_size // 4)))),
        })

    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_saved_structure(index_path, idx_type):
    with np.load(index_path, allow_pickle=False) as data:
        structure = {name: data[name] for name in data.files}
    if idx_type == 'hnsw':
        structure['entry_point'] = int(structure['entry_point'][0])
        structure['max_level'] = int(structure['max_level'][0])
    return structure


def load_saved_id_map(id_map_path):
    with open(id_map_path) as f:
        return json.load(f)


class QueryVideoStore:
    def __init__(self, cache_dir, dataset, id_map):
        self.cache_dir = resolve_dataset_cache_dir(cache_dir, dataset)
        self.id_map = id_map
        self.cache = {}
        self.video_load_s = 0.0
        self.video_pool_s = 0.0
        self.disk_reads = 0

    def get(self, row_id):
        row_id = int(row_id)
        cached = self.cache.get(row_id)
        if cached is not None:
            return cached

        video_id = normalize_cache_video_id(self.id_map[row_id])
        cache_path = os.path.join(self.cache_dir, f"{video_id}.npz")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"Cached video features not found: {cache_path}")

        t0 = time.perf_counter()
        with np.load(cache_path) as data:
            frame_embeds = data['frame_embeds']
        self.video_load_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        vector = pooled_video_vector(frame_embeds)
        self.video_pool_s += time.perf_counter() - t0

        self.cache[row_id] = vector
        self.disk_reads += 1
        return vector


def finalize_topk(topk_heap, k):
    retrieved = np.full(k, -1, dtype=np.int64)
    distances = np.full(k, -np.inf, dtype=np.float32)
    ordered = sorted(topk_heap, key=lambda item: (-item[0], item[1]))
    for i, (score, row_id) in enumerate(ordered[:k]):
        distances[i] = np.float32(score)
        retrieved[i] = np.int64(row_id)
    return retrieved, distances


def topk_desc(scores, k):
    if scores.size == 0:
        return np.empty((0,), dtype=np.int64)
    k = min(k, scores.size)
    if k <= 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def hnsw_neighbors(structure, node_id, level):
    if level >= int(structure['levels'][node_id]):
        return np.empty((0,), dtype=np.int32)
    base = int(structure['offsets'][node_id])
    start = base + int(structure['cum_nneighbor_per_level'][level])
    end = base + int(structure['cum_nneighbor_per_level'][level + 1])
    neighbors = structure['neighbors'][start:end]
    return neighbors[neighbors >= 0]


def search_manual_ivf(query_vec, structure, args, vector_store, k):
    similarity_s = 0.0
    topk_heap = []
    centroids = structure['centroids']
    centroid_scores = centroids @ query_vec
    probed_lists = topk_desc(centroid_scores, min(args.ivf_nprobe, centroids.shape[0]))
    list_offsets = structure['list_offsets']
    list_row_ids = structure['list_row_ids']

    scored = 0
    for list_no in probed_lists:
        start = int(list_offsets[list_no])
        end = int(list_offsets[list_no + 1])
        for row_id in list_row_ids[start:end]:
            vector = vector_store.get(row_id)
            t0 = time.perf_counter()
            score = float(np.dot(query_vec, vector))
            similarity_s += time.perf_counter() - t0
            scored += 1
            item = (score, int(row_id))
            if len(topk_heap) < k:
                heapq.heappush(topk_heap, item)
            elif score > topk_heap[0][0]:
                heapq.heapreplace(topk_heap, item)

    retrieved, distances = finalize_topk(topk_heap, k)
    return retrieved, distances, {
        'similarity_s': similarity_s,
        'vectors_scored': scored,
        'lists_probed': int(len(probed_lists)),
    }


def search_manual_hnsw(query_vec, structure, args, vector_store, k):
    similarity_s = 0.0
    score_cache = {}

    def score_node(row_id):
        row_id = int(row_id)
        cached = score_cache.get(row_id)
        if cached is not None:
            return cached
        vector = vector_store.get(row_id)
        t0 = time.perf_counter()
        score = float(np.dot(query_vec, vector))
        nonlocal similarity_s
        similarity_s += time.perf_counter() - t0
        score_cache[row_id] = score
        return score

    entry_point = int(structure['entry_point'])
    if entry_point < 0:
        return finalize_topk([], k) + ({
            'similarity_s': 0.0,
            'vectors_scored': 0,
            'visited_nodes': 0,
        },)

    current = entry_point
    current_score = score_node(current)
    for level in range(int(structure['max_level']), 0, -1):
        improved = True
        while improved:
            improved = False
            for neighbor in hnsw_neighbors(structure, current, level):
                neighbor = int(neighbor)
                neighbor_score = score_node(neighbor)
                if neighbor_score > current_score:
                    current = neighbor
                    current_score = neighbor_score
                    improved = True

    ef_search = max(k, min(args.hnsw_ef_search, len(structure['levels'])))
    visited = {current}
    candidates = [(-current_score, current)]
    topk_heap = [(current_score, current)]

    while candidates:
        neg_score, node_id = heapq.heappop(candidates)
        node_score = -neg_score
        lower_bound = topk_heap[0][0]
        if len(topk_heap) >= ef_search and node_score < lower_bound:
            break
        for neighbor in hnsw_neighbors(structure, int(node_id), 0):
            neighbor = int(neighbor)
            if neighbor in visited:
                continue
            visited.add(neighbor)
            neighbor_score = score_node(neighbor)
            if len(topk_heap) < ef_search or neighbor_score > topk_heap[0][0]:
                heapq.heappush(candidates, (-neighbor_score, neighbor))
                heapq.heappush(topk_heap, (neighbor_score, neighbor))
                if len(topk_heap) > ef_search:
                    heapq.heappop(topk_heap)

    retrieved, distances = finalize_topk(topk_heap, k)
    return retrieved, distances, {
        'similarity_s': similarity_s,
        'vectors_scored': len(score_cache),
        'visited_nodes': len(visited),
    }


def search_manual(query_vec, structure, idx_type, args, vector_store, k):
    if idx_type == 'ivf':
        return search_manual_ivf(query_vec, structure, args, vector_store, k)
    if idx_type == 'hnsw':
        return search_manual_hnsw(query_vec, structure, args, vector_store, k)
    raise ValueError(f"Unsupported ANN index type: {idx_type}")


# Evaluation
def compute_metrics(ranks):
    return {
        'R@1': 100.0 * float(np.mean(ranks < 1)),
        'R@5': 100.0 * float(np.mean(ranks < 5)),
        'R@10': 100.0 * float(np.mean(ranks < 10)),
        'MedR': float(np.median(ranks) + 1),
        'MeanR': float(np.mean(ranks) + 1),
    }


def search_batched(query_embs, structure, id_map, idx_type, args, gt_indices, k):
    """Throughput path: query embeddings are ready; search one query at a time."""
    query_embs = normalize(query_embs)
    n = len(query_embs)
    retrieved = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), -np.inf, dtype=np.float32)
    total_search_s = 0.0
    for i in tqdm(range(n), desc="Searching queries"):
        vector_store = QueryVideoStore(args.cache_dir, args.dataset, id_map)
        t0 = time.perf_counter()
        rows, scores, _ = search_manual(
            query_embs[i], structure, idx_type, args, vector_store, k)
        total_search_s += time.perf_counter() - t0
        retrieved[i] = rows
        distances[i] = scores

    ranks = np.full(n, k + 1, dtype=np.float64)
    for i in range(n):
        if gt_indices[i] < 0:
            continue
        m = np.where(retrieved[i] == gt_indices[i])[0]
        if len(m) > 0:
            ranks[i] = m[0]
    metrics = compute_metrics(ranks)
    metrics['search_time_total_s'] = float(total_search_s)
    metrics['search_time_per_query_ms'] = 1000.0 * total_search_s / max(n, 1)
    metrics['queries_per_sec'] = n / total_search_s if total_search_s > 0 else 0
    return metrics, retrieved, distances, None  # last slot reserved for per-query timings


def search_per_query(captions, model, tokenizer, device, index_path, id_map, idx_type,
                     args, gt_indices, k, num_warmup):
    """Online path: preload structure once; reload video features from disk per query."""
    n = len(captions)
    retrieved = np.full((n, k), -1, dtype=np.int64)
    distances = np.full((n, k), -np.inf, dtype=np.float32)
    timings = []

    print(f"  Warmup text encoder only: {num_warmup} queries (excluded from timing)")
    for i in range(min(num_warmup, n)):
        _ = encode_text_per_query(captions[i], model, tokenizer, device)

    structure_load_start = time.perf_counter()
    structure = load_saved_structure(index_path, idx_type)
    structure_load_s = time.perf_counter() - structure_load_start

    print(f"  Per-query timing: {n} queries "
          f"(batch=1 encode + disk video loads + batch=1 search)")
    for i in tqdm(range(n), desc="Per-query"):
        t_total = time.perf_counter()
        emb, t_enc = encode_text_per_query(captions[i], model, tokenizer, device)
        vector_store = QueryVideoStore(args.cache_dir, args.dataset, id_map)
        query_vec = normalize(emb)[0]
        t0 = time.perf_counter()
        rows, scores, search_stats = search_manual(
            query_vec, structure, idx_type, args, vector_store, k)
        t_search = time.perf_counter() - t0
        retrieved[i] = rows
        distances[i] = scores
        timings.append({
            'encode_s': t_enc,
            'index_load_s': 0.0,
            'video_load_s': vector_store.video_load_s,
            'video_pool_s': vector_store.video_pool_s,
            'similarity_s': search_stats['similarity_s'],
            'search_s': t_search,
            'total_s': time.perf_counter() - t_total,
            'vectors_scored': search_stats.get('vectors_scored', 0),
            'lists_probed': search_stats.get('lists_probed', 0),
            'visited_nodes': search_stats.get('visited_nodes', 0),
            'disk_reads': vector_store.disk_reads,
        })

    ranks = np.full(n, k + 1, dtype=np.float64)
    for i in range(n):
        if gt_indices[i] < 0:
            continue
        m = np.where(retrieved[i] == gt_indices[i])[0]
        if len(m) > 0:
            ranks[i] = m[0]
    metrics = compute_metrics(ranks)
    enc_arr = np.array([t['encode_s'] for t in timings])
    load_arr = np.array([t['index_load_s'] for t in timings])
    video_load_arr = np.array([t['video_load_s'] for t in timings])
    video_pool_arr = np.array([t['video_pool_s'] for t in timings])
    similarity_arr = np.array([t['similarity_s'] for t in timings])
    sea_arr = np.array([t['search_s'] for t in timings])
    total_arr = np.array([t['total_s'] for t in timings])
    metrics['encode_time_per_query_ms_mean'] = float(1000.0 * enc_arr.mean())
    metrics['encode_time_per_query_ms_std'] = float(1000.0 * enc_arr.std())
    metrics['index_load_time_per_query_ms_mean'] = float(1000.0 * load_arr.mean())
    metrics['index_load_time_per_query_ms_std'] = float(1000.0 * load_arr.std())
    metrics['video_load_time_per_query_ms_mean'] = float(1000.0 * video_load_arr.mean())
    metrics['video_load_time_per_query_ms_std'] = float(1000.0 * video_load_arr.std())
    metrics['video_pool_time_per_query_ms_mean'] = float(1000.0 * video_pool_arr.mean())
    metrics['video_pool_time_per_query_ms_std'] = float(1000.0 * video_pool_arr.std())
    metrics['similarity_time_per_query_ms_mean'] = float(1000.0 * similarity_arr.mean())
    metrics['similarity_time_per_query_ms_std'] = float(1000.0 * similarity_arr.std())
    metrics['search_time_per_query_ms_mean'] = float(1000.0 * sea_arr.mean())
    metrics['search_time_per_query_ms_std'] = float(1000.0 * sea_arr.std())
    metrics['online_time_per_query_ms_mean'] = float(1000.0 * total_arr.mean())
    metrics['online_time_per_query_ms_std'] = float(1000.0 * total_arr.std())
    metrics['search_time_total_s'] = float(sea_arr.sum())
    metrics['online_time_total_s'] = float(total_arr.sum())
    metrics['queries_per_sec'] = float(n / max(total_arr.sum(), 1e-9))
    metrics['structure_load_time_s'] = float(structure_load_s)
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
                'index_load': 1000.0 * t['index_load_s'],
                'video_load': 1000.0 * t['video_load_s'],
                'video_pool': 1000.0 * t['video_pool_s'],
                'similarity': 1000.0 * t['similarity_s'],
                'search': 1000.0 * t['search_s'],
                'total': 1000.0 * t['total_s'],
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
            'index_load_mean': metrics.get('index_load_time_per_query_ms_mean'),
            'index_load_std': metrics.get('index_load_time_per_query_ms_std'),
            'video_load_mean': metrics.get('video_load_time_per_query_ms_mean'),
            'video_load_std': metrics.get('video_load_time_per_query_ms_std'),
            'video_pool_mean': metrics.get('video_pool_time_per_query_ms_mean'),
            'video_pool_std': metrics.get('video_pool_time_per_query_ms_std'),
            'similarity_mean': metrics.get('similarity_time_per_query_ms_mean'),
            'similarity_std': metrics.get('similarity_time_per_query_ms_std'),
            'search_mean': metrics.get('search_time_per_query_ms_mean'),
            'search_std': metrics.get('search_time_per_query_ms_std'),
            'online_total_mean': metrics.get('online_time_per_query_ms_mean'),
            'online_total_std': metrics.get('online_time_per_query_ms_std'),
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
    if args.index_dir is None:
        args.index_dir = os.path.join(args.output_dir, 'indexes')

    if args.num_candidates is None:
        args.num_candidates = ANN_BASELINE_NUM_CANDIDATES[(args.dataset, args.setting)]
        print(f"Using ANN baseline K = {args.num_candidates} for ({args.dataset}, t{args.setting})")

    print(f"\n{'='*70}")
    print(f"ANN Dense Retrieval Baseline")
    print(f"Dataset: {args.dataset}, Setting: {args.setting}, Index: {args.index_type}, K={args.num_candidates}")
    print(f"Per-query timing: {args.per_query_timing}")
    print(f"{'='*70}\n")

    # 1. Load test queries (raw IDs).
    if args.query_manifest:
        test_pairs = load_query_manifest(args.query_manifest)
        print(f"Loaded query manifest: {args.query_manifest}")
    else:
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
    t0 = time.perf_counter()
    video_embs, valid_pool_ids = load_video_embeddings(
        pool_video_ids, args.cache_dir, args.dataset)
    video_embedding_load_time_s = time.perf_counter() - t0
    print(f"Loaded {len(valid_pool_ids)} video embeddings, dim={video_embs.shape[1]}")
    print(f"Video embeddings loaded in {video_embedding_load_time_s:.2f}s")

    vid_to_pool_idx = {vid: i for i, vid in enumerate(valid_pool_ids)}

    # 4. Build GT indices using base-stripped query video ids. Keep every
    # query in the export; if its GT video is absent from the ANN pool, the
    # query remains in the metric denominator and is evaluated as a miss.
    gt_indices, gt_vids = [], []
    for vid, _ in test_pairs:
        base = strip_clip_suffix(vid)
        gt_indices.append(vid_to_pool_idx.get(base, -1))
        gt_vids.append(base)
    gt_indices = np.array(gt_indices)
    gt_in_pool = int(np.sum(gt_indices >= 0))
    print(f"Queries with GT in pool: {gt_in_pool}/{len(test_pairs)}")

    # 5. Load CLIP weights from XPool checkpoint.
    print(f"\nLoading X-Pool checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint['state_dict']
    clip_state_dict = {k[5:]: v for k, v in state_dict.items() if k.startswith('clip.')}
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.load_state_dict(clip_state_dict, strict=False)
    clip_model = clip_model.to(device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Throughput-mode pre-encoding (only when not in per-query timing mode).
    if not args.per_query_timing:
        query_embs = encode_text_queries_batched(
            test_captions, clip_model, tokenizer, device, args.batch_size)
        print(f"Encoded {len(test_captions)} text queries, dim={query_embs.shape[1]}")

    # 6. Build indices and evaluate.
    search_k = max(100, args.num_candidates)
    index_types = list(SUPPORTED_INDEX_TYPES) if args.index_type == 'all' else [args.index_type]
    all_results = {}
    index_artifacts = {}

    for idx_type in index_types:
        print(f"\n--- {idx_type.upper()} Index ---")
        t0 = time.perf_counter()
        index = build_index(video_embs, idx_type, args)
        build_time = time.perf_counter() - t0
        print(f"Index built in {build_time:.2f}s ({index.ntotal} vectors)")
        index_path = index_path_for(args.index_dir, idx_type, args.dataset, args.setting)
        meta_path = index_meta_path_for(args.index_dir, idx_type, args.dataset, args.setting)
        id_map_path = index_id_map_path_for(args.index_dir, idx_type, args.dataset, args.setting)
        save_index_artifacts(
            index,
            index_path,
            meta_path,
            id_map_path,
            idx_type,
            args,
            valid_pool_ids,
            video_embedding_load_time_s,
            build_time,
        )
        print(f"Offline structure saved to: {index_path}")
        del index
        saved_pool_ids = load_saved_id_map(id_map_path)
        index_artifacts[idx_type] = {
            'structure_path': index_path,
            'metadata_path': meta_path,
            'id_map_path': id_map_path,
        }

        if args.per_query_timing:
            metrics, retrieved, distances, per_query_timings = search_per_query(
                test_captions, clip_model, tokenizer, device, index_path,
                saved_pool_ids,
                idx_type, args, gt_indices, search_k, args.num_warmup)
        else:
            t0 = time.perf_counter()
            structure = load_saved_structure(index_path, idx_type)
            structure_load_time_s = time.perf_counter() - t0
            metrics, retrieved, distances, per_query_timings = search_batched(
                query_embs, structure, saved_pool_ids, idx_type, args, gt_indices, search_k)
            metrics['index_load_time_s'] = float(structure_load_time_s)
            metrics['structure_load_time_s'] = float(structure_load_time_s)

        metrics['video_embedding_load_time_s'] = float(video_embedding_load_time_s)
        metrics['build_time_s'] = build_time
        all_results[idx_type] = metrics

        print(f"R@1: {metrics['R@1']:.2f}  R@5: {metrics['R@5']:.2f}  "
              f"R@10: {metrics['R@10']:.2f}  MedR: {metrics['MedR']:.1f}  "
              f"MeanR: {metrics['MeanR']:.1f}")
        if args.per_query_timing:
            print(f"Encode/q: {metrics['encode_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['encode_time_per_query_ms_std']:.2f} ms; "
                  f"Video load/q: {metrics['video_load_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['video_load_time_per_query_ms_std']:.2f} ms; "
                  f"Search/q: {metrics['search_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['search_time_per_query_ms_std']:.2f} ms; "
                  f"Online total/q: {metrics['online_time_per_query_ms_mean']:.2f} +/- "
                  f"{metrics['online_time_per_query_ms_std']:.2f} ms")
        else:
            print(f"Search: {metrics['search_time_total_s']:.4f}s "
                  f"({metrics['queries_per_sec']:.0f} q/s), "
                  f"Build: {build_time:.2f}s, "
                  f"Structure load: {metrics['structure_load_time_s']:.4f}s")

        cand_filename = (f"{args.dataset}_ann_{idx_type}"
                         f"_{args.num_candidates}_candidates"
                         f"_t{args.setting}.json")
        cand_path = os.path.join(args.candidate_dir, cand_filename)
        save_candidate_json(retrieved, distances, test_captions, gt_vids,
                            saved_pool_ids, args.num_candidates, metrics, args,
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
            'num_queries': len(test_captions),
            'pool_size': len(valid_pool_ids),
            'video_embedding_load_time_s': float(video_embedding_load_time_s),
            'per_query_timing': args.per_query_timing,
            'index_artifacts': index_artifacts,
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
                        'index_load_ms_mean', 'index_load_ms_std',
                        'video_load_ms_mean', 'video_load_ms_std',
                        'video_pool_ms_mean', 'video_pool_ms_std',
                        'similarity_ms_mean', 'similarity_ms_std',
                        'search_ms_mean', 'search_ms_std',
                        'online_total_ms_mean', 'online_total_ms_std',
                        'video_load_s', 'build_s'])
            for idx_type, m in all_results.items():
                w.writerow([idx_type, f"{m['R@1']:.2f}", f"{m['R@5']:.2f}",
                            f"{m['R@10']:.2f}", f"{m['MedR']:.1f}",
                            f"{m['MeanR']:.1f}",
                            f"{m['encode_time_per_query_ms_mean']:.3f}",
                            f"{m['encode_time_per_query_ms_std']:.3f}",
                            f"{m['index_load_time_per_query_ms_mean']:.3f}",
                            f"{m['index_load_time_per_query_ms_std']:.3f}",
                            f"{m['video_load_time_per_query_ms_mean']:.3f}",
                            f"{m['video_load_time_per_query_ms_std']:.3f}",
                            f"{m['video_pool_time_per_query_ms_mean']:.3f}",
                            f"{m['video_pool_time_per_query_ms_std']:.3f}",
                            f"{m['similarity_time_per_query_ms_mean']:.3f}",
                            f"{m['similarity_time_per_query_ms_std']:.3f}",
                            f"{m['search_time_per_query_ms_mean']:.3f}",
                            f"{m['search_time_per_query_ms_std']:.3f}",
                            f"{m['online_time_per_query_ms_mean']:.3f}",
                            f"{m['online_time_per_query_ms_std']:.3f}",
                            f"{m['video_embedding_load_time_s']:.2f}",
                            f"{m['build_time_s']:.2f}"])
        else:
            w.writerow(['index_type', 'R@1', 'R@5', 'R@10', 'MedR', 'MeanR',
                        'index_load_s', 'search_time_s', 'video_load_s', 'build_s'])
            for idx_type, m in all_results.items():
                w.writerow([idx_type, f"{m['R@1']:.2f}", f"{m['R@5']:.2f}",
                            f"{m['R@10']:.2f}", f"{m['MedR']:.1f}",
                            f"{m['MeanR']:.1f}",
                            f"{m['index_load_time_s']:.4f}",
                            f"{m['search_time_total_s']:.4f}",
                            f"{m['video_embedding_load_time_s']:.2f}",
                            f"{m['build_time_s']:.2f}"])
    print(f"CSV saved to: {csv_file}")


if __name__ == '__main__':
    main()
