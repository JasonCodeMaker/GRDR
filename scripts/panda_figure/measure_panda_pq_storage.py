#!/usr/bin/env python
"""Measure Panda 2M IVF-PQ/OPQ storage from actual FAISS serialized indexes.

This is storage-only: it reuses the Panda pool-scaling manifest, builds FAISS
IVF-PQ / OPQ indexes with the Fig. 2 configuration, and records
faiss.serialize_index(index).nbytes. It does not encode queries or run rerank.

The default storage-equivalent mode does not read 2M Panda .npz files. FAISS
serialized index length for these baselines depends on N, dim, nlist, PQ m/nbits,
and index type; feature values change code contents, not the number of stored
codes, ids, centroids, or transforms.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np


REPO = Path("/home/uqzzha35/Project/SemanticID/GRDR")
DEFAULT_MANIFEST = REPO / "var/research/2026-06-01-panda-pool-scaling/manifests/panda_pool_d2000000.json"
DEFAULT_TEST_JSON = REPO / "data/panda/video_retreival_caption/panda_ret_test.json"
DEFAULT_CACHE = REPO / "reranker/xpool/video_features_cache/Xpool-Panda"
DEFAULT_OUT = REPO / "var/research/2026-06-22-panda-pq-storage"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--test-json", default=str(DEFAULT_TEST_JSON))
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT))
    p.add_argument("--methods", nargs="+", default=["ivfpq", "opq"], choices=["ivfpq", "opq"])
    p.add_argument("--pq-m", type=int, default=16)
    p.add_argument("--pq-nbits", type=int, default=8)
    p.add_argument("--ivf-nlist", type=int, default=4096)
    p.add_argument("--ivf-nprobe", type=int, default=256)
    p.add_argument("--faiss-threads", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--storage-equivalent", action="store_true", default=True,
                   help="Build a byte-equivalent FAISS artifact without reading Panda feature files.")
    p.add_argument("--read-panda-features", dest="storage_equivalent", action="store_false",
                   help="Read actual Panda feature files before building the index; slow for 2M.")
    p.add_argument("--add-batch-size", type=int, default=100_000)
    p.add_argument("--max-vectors", type=int, default=None,
                   help="Optional smoke-test cap after pool construction.")
    p.add_argument("--force-embeddings", action="store_true",
                   help="Rebuild the pooled embedding memmap even if it already exists.")
    p.add_argument("--force-index", action="store_true",
                   help="Rebuild an index even if its measured JSON already exists.")
    return p.parse_args()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bare_video_id(video_path):
    return video_path.replace(".mp4", "").split("/")[-1]


def load_pool_ids(test_json, manifest, max_vectors=None):
    with open(test_json) as f:
        test_rows = json.load(f)
    test_ids = []
    seen = set()
    for row in test_rows:
        vid = bare_video_id(row["video"])
        if vid not in seen:
            seen.add(vid)
            test_ids.append(vid)

    with open(manifest) as f:
        man = json.load(f)
    pool = list(test_ids)
    test_set = set(test_ids)
    for vid in man["video_ids"]:
        bare = bare_video_id(vid)
        if bare not in test_set:
            pool.append(bare)
    if max_vectors is not None:
        pool = pool[:max_vectors]
    return pool, test_set, man


def load_manifest(manifest):
    with open(manifest) as f:
        return json.load(f)


def pooled_video_vector(npz_path):
    with np.load(npz_path) as data:
        if "frame_embeds" in data:
            pooled = data["frame_embeds"].mean(axis=0).astype(np.float32, copy=False)
        elif "video_embed" in data:
            pooled = data["video_embed"].astype(np.float32, copy=False)
        else:
            raise KeyError(f"{npz_path} has neither frame_embeds nor video_embed")
    norm = np.linalg.norm(pooled)
    if norm > 1e-8:
        pooled = pooled / norm
    return pooled.astype(np.float32, copy=False)


def cache_path(cache_dir, vid, test_set):
    split = "test" if vid in test_set else "train"
    return Path(cache_dir) / "PANDA" / split / f"{vid}.npz"


def build_embedding_memmap(pool_ids, test_set, cache_dir, out_dir, force=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{len(pool_ids)}x512"
    mmap_path = out_dir / f"panda_pool_embeddings_{suffix}.float32.mmap"
    meta_path = out_dir / f"panda_pool_embeddings_{suffix}.meta.json"

    if mmap_path.exists() and meta_path.exists() and not force:
        meta = json.load(open(meta_path))
        emb = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(meta["valid_count"], meta["dim"]))
        print(f"reusing embeddings: {mmap_path} shape={emb.shape}")
        return emb, meta

    first = None
    for vid in pool_ids:
        path = cache_path(cache_dir, vid, test_set)
        if path.exists():
            first = pooled_video_vector(path)
            break
    if first is None:
        raise RuntimeError("No feature files found for the requested Panda pool")

    dim = int(first.shape[0])
    emb = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(len(pool_ids), dim))
    valid_ids = []
    missing = []
    t0 = time.perf_counter()
    for i, vid in enumerate(pool_ids):
        path = cache_path(cache_dir, vid, test_set)
        if not path.exists():
            missing.append(vid)
            continue
        emb[len(valid_ids)] = pooled_video_vector(path)
        valid_ids.append(vid)
        if len(valid_ids) % 50_000 == 0:
            elapsed = time.perf_counter() - t0
            print(f"loaded {len(valid_ids):,}/{len(pool_ids):,} embeddings in {elapsed/60:.1f} min", flush=True)
    emb.flush()

    meta = {
        "mmap_path": str(mmap_path),
        "requested_count": len(pool_ids),
        "valid_count": len(valid_ids),
        "missing_count": len(missing),
        "dim": dim,
        "created_s": time.perf_counter() - t0,
        "missing_examples": missing[:20],
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    emb = np.memmap(mmap_path, dtype=np.float32, mode="r", shape=(len(valid_ids), dim))
    print(f"built embeddings: {mmap_path} shape={emb.shape}")
    return emb, meta


def build_index(embeddings, method, nlist, nprobe, pq_m, pq_nbits):
    n, dim = embeddings.shape
    nlist = min(nlist, max(1, n // 4))
    nprobe = min(nprobe, nlist)
    if method == "ivfpq":
        index = faiss.index_factory(dim, f"IVF{nlist},PQ{pq_m}x{pq_nbits}", faiss.METRIC_INNER_PRODUCT)
    elif method == "opq":
        index = faiss.index_factory(dim, f"OPQ{pq_m},IVF{nlist},PQ{pq_m}x{pq_nbits}", faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(method)

    t0 = time.perf_counter()
    index.train(embeddings)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    index.add(embeddings)
    add_s = time.perf_counter() - t0
    faiss.extract_index_ivf(index).nprobe = nprobe
    return index, {"train_s": train_s, "add_s": add_s, "nlist": nlist, "nprobe": nprobe}


def _initialized_ivfpq(dim, nlist, pq_m, pq_nbits, seed):
    """Create a valid trained IVF-PQ shell for storage measurement.

    The centroid/codebook values are placeholders. They affect recall, not the
    serialized byte length. A small smoke test against trained real Panda vectors
    verifies the same byte count for identical N/configuration.
    """
    rng = np.random.default_rng(seed)
    index = faiss.index_factory(dim, f"IVF{nlist},PQ{pq_m}x{pq_nbits}", faiss.METRIC_INNER_PRODUCT)
    ivf = faiss.downcast_index(faiss.extract_index_ivf(index))

    centroids = rng.standard_normal((nlist, dim)).astype(np.float32)
    faiss.normalize_L2(centroids)
    ivf.quantizer.add(centroids)

    pq_centroids = np.zeros(ivf.pq.centroids.size(), dtype=np.float32)
    faiss.copy_array_to_vector(pq_centroids, ivf.pq.centroids)
    ivf.is_trained = True
    index.is_trained = True
    return index


def build_storage_equivalent_index(total_n, dim, method, nlist, nprobe, pq_m, pq_nbits, batch_size):
    nlist = min(nlist, max(1, total_n // 4))
    nprobe = min(nprobe, nlist)
    t0 = time.perf_counter()
    if method == "ivfpq":
        index = _initialized_ivfpq(dim, nlist, pq_m, pq_nbits, seed=20260622)
    elif method == "opq":
        base = _initialized_ivfpq(dim, nlist, pq_m, pq_nbits, seed=20260623)
        opq = faiss.OPQMatrix(dim, pq_m)
        identity = np.eye(dim, dtype=np.float32).ravel()
        faiss.copy_array_to_vector(identity, opq.A)
        opq.is_trained = True
        index = faiss.IndexPreTransform(opq, base)
        index.is_trained = True
    else:
        raise ValueError(method)
    init_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    remaining = int(total_n)
    rng = np.random.default_rng(20260624)
    while remaining > 0:
        this_n = min(batch_size, remaining)
        batch = rng.standard_normal((this_n, dim)).astype(np.float32)
        faiss.normalize_L2(batch)
        index.add(batch)
        remaining -= this_n
    add_s = time.perf_counter() - t0
    faiss.extract_index_ivf(index).nprobe = nprobe
    return index, {
        "init_s": init_s,
        "add_s": add_s,
        "nlist": nlist,
        "nprobe": nprobe,
        "storage_equivalent": True,
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(args.faiss_threads)

    if args.storage_equivalent:
        manifest = load_manifest(args.manifest)
        measured_pool_size = int(args.max_vectors or manifest["pool_size"])
        embeddings = None
        emb_meta = {
            "feature_source": "storage_equivalent_values",
            "requested_count": measured_pool_size,
            "valid_count": measured_pool_size,
            "dim": int(args.dim),
            "note": (
                "IVF-PQ/OPQ serialized byte length is independent of Panda feature "
                "values for fixed N, dim, nlist, PQ m/nbits, and FAISS index type."
            ),
        }
    else:
        pool_ids, test_set, manifest = load_pool_ids(args.test_json, args.manifest, args.max_vectors)
        measured_pool_size = len(pool_ids)
        embeddings, emb_meta = build_embedding_memmap(
            pool_ids, test_set, args.cache_dir, out_dir, force=args.force_embeddings)

    print(f"pool requested={measured_pool_size:,} manifest_pool={manifest.get('pool_size')} seed={manifest.get('seed')}", flush=True)
    print(f"faiss_threads={args.faiss_threads} methods={args.methods}")

    common = {
        "dataset": "panda",
        "distractor_n": manifest.get("n_distractors"),
        "manifest_pool_size": manifest.get("pool_size"),
        "measured_pool_size": int(measured_pool_size),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "cache_dir": str(args.cache_dir),
        "feature_backbone": "xpool_clip",
        "dim": int(args.dim if embeddings is None else embeddings.shape[1]),
        "pq_m": int(args.pq_m),
        "pq_nbits": int(args.pq_nbits),
        "ivf_nlist_requested": int(args.ivf_nlist),
        "ivf_nprobe_requested": int(args.ivf_nprobe),
        "faiss_threads": int(args.faiss_threads),
        "embedding_meta": emb_meta,
        "storage_contract": "faiss.serialize_index(index).nbytes",
    }

    for method in args.methods:
        result_path = out_dir / f"panda_d{manifest.get('n_distractors')}_{method}_m{args.pq_m}_storage.json"
        index_path = out_dir / f"panda_d{manifest.get('n_distractors')}_{method}_m{args.pq_m}.faiss"
        if result_path.exists() and not args.force_index:
            print(f"skip existing {result_path}")
            continue

        print(f"building {method}...", flush=True)
        t0 = time.perf_counter()
        if args.storage_equivalent:
            index, timings = build_storage_equivalent_index(
                measured_pool_size,
                args.dim,
                method,
                args.ivf_nlist,
                args.ivf_nprobe,
                args.pq_m,
                args.pq_nbits,
                args.add_batch_size,
            )
        else:
            index, timings = build_index(
                embeddings, method, args.ivf_nlist, args.ivf_nprobe, args.pq_m, args.pq_nbits)
        build_s = time.perf_counter() - t0
        index_bytes = int(faiss.serialize_index(index).nbytes)
        faiss.write_index(index, str(index_path))
        file_bytes = index_path.stat().st_size
        result = {
            **common,
            "method": method,
            "index_ntotal": int(index.ntotal),
            "faiss_index_bytes": index_bytes,
            "written_index_bytes": int(file_bytes),
            "bytes_per_video": float(index_bytes / max(index.ntotal, 1)),
            "build_s": build_s,
            **timings,
            "index_path": str(index_path),
        }
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps({
            "method": method,
            "faiss_index_bytes": index_bytes,
            "bytes_per_video": result["bytes_per_video"],
            "build_s": build_s,
            "result_path": str(result_path),
        }, indent=2), flush=True)
        del index


if __name__ == "__main__":
    main()
