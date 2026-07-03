#!/usr/bin/env python
"""Measure index storage vs corpus size N over the panda corpora to 2M (Fig 2).

Supersedes reranker/xpool/utils/storage_sim.py (which extrapolated from random
features). GRDR follows the project Storage Accounting SSOT; the baselines are
measured byte counts:

  - GRDR        : per-video index footprint per the Storage Accounting SSOT
                 (scripts/panda_figure/storage_accounting_ssot.md): V*L sID codes +
                 int64 video id = 32 B/vid. The T5 decoder, RQ-VAE codebooks, and the
                 prefix trie are EXCLUDED (query-side / offline / not persisted).
  - frame dense (X-Pool)   : measured per-video .npz bytes x N.
  - video dense (CLIP4Clip): measured per-video .npz bytes x N.
  - HNSW        : same measured video-feature anchor as CLIP4Clip + measured persisted
                  graph artifact bytes/video.
  - IVF-PQ m=16 : FAISS serialized-index footprint: PQ codes + vector ids + PQ
                  codebook + IVF coarse centroids + measured FAISS metadata.
  - OPQ m=16    : IVF-PQ m=16 footprint + OPQ rotation + measured FAISS metadata.
  - IVF-Flat    : raw vectors (N x dim x 4B) + int64 id; kept in the CSV for audit but
                  omitted from the Fig. 2 storage panel because its storage strategy is
                  the same video-feature anchor as CLIP4Clip.

Writes a long CSV (method, N, component, bytes) so the notebook renders directly.
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from storage_ssot import (
    GRDR_V,
    GRDR_L,
    GRDR_B_CODE,
    GRDR_B_ID,
    ANN_B_ID,
    IVF_NLIST,
    PQ_M,
    PQ_NBITS,
    IVFPQ_FAISS_META_BYTES,
    OPQ_FAISS_META_BYTES,
    grdr_bytes_per_video,
    hnsw_bytes_per_video,
    ivf_coarse_centroid_bytes,
    ivfpq_bytes,
    opq_bytes,
    opq_rotation_bytes,
    pq_codebook_bytes,
    measured_hnsw_graph_bytes_per_video,
)

REPO = "/home/uqzzha35/Project/SemanticID/GRDR"

# Measured panda corpus grid (test=5694; +d distractors).
PANDA_TEST = 5694
DISTRACTORS = [0, 400_000, 800_000, 1_200_000, 1_600_000, 2_000_000]
NS = [PANDA_TEST + d for d in DISTRACTORS]

DIM = 512  # InternVideo2 / CLIP video embedding dim.

# GRDR per-video footprint = V*L sID codes + int64 id (constants from storage_ssot.py,
# the single importable SSOT; see storage_accounting_ssot.md).

# --- measured baseline artifact paths -----------------------------------------
# Per-video dense feature cache anchors (measured on disk).
FRAME_CACHE = f"{REPO}/reranker/xpool/video_features_cache/Xpool-Panda"  # 28 GB / 2.15M
VIDEO_CACHE = f"{REPO}/reranker/xpool/video_features_cache/CLIP4clip"
HNSW_INDEX_DIR = f"{REPO}/output/evaluation_results/figures/ann_baseline/indexes"


def measured_per_video_bytes(cache_dir, n_sample=400):
    """Mean .npz bytes over a sample of cached feature files."""
    sizes = []
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if f.endswith(".npz"):
                sizes.append(os.path.getsize(os.path.join(root, f)))
            if len(sizes) >= n_sample:
                break
        if len(sizes) >= n_sample:
            break
    return float(np.mean(sizes)) if sizes else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REPO}/output/evaluation_results/figures/summaries/storage_data.csv")
    args = ap.parse_args()

    # --- per-unit measured quantities (baselines) ---
    frame_pv = measured_per_video_bytes(FRAME_CACHE)           # frame dense slope
    video_pv = measured_per_video_bytes(VIDEO_CACHE)           # video dense slope
    hnsw_graph_pv = measured_hnsw_graph_bytes_per_video(HNSW_INDEX_DIR)
    vec_pv = DIM * 4                                           # raw f32 vector / video
    grdr_pv = grdr_bytes_per_video()                          # SSOT 32 B/video
    anchor_pv = vec_pv + ANN_B_ID                             # vector + int64 id
    hnsw_pv = hnsw_bytes_per_video(video_pv, hnsw_graph_pv)
    pq_codebook_b = pq_codebook_bytes(DIM, PQ_M, PQ_NBITS)
    ivf_centroid_b = ivf_coarse_centroid_bytes(DIM, IVF_NLIST)
    opq_rotation_b = opq_rotation_bytes(DIM)

    print(f"  GRDR bytes/video   : {grdr_pv} B  (SSOT: {GRDR_V}*{GRDR_L}*{GRDR_B_CODE} codes + {GRDR_B_ID} id)")
    print(f"  frame dense/video  : {frame_pv/1024:.2f} KB")
    print(f"  video dense/video  : {video_pv/1024:.2f} KB")
    print(f"  raw dense anchor/video : {anchor_pv} B  ({vec_pv} B vector + {ANN_B_ID} B id; IVF audit only)")
    print(f"  HNSW graph/video   : {hnsw_graph_pv:.2f} B  (measured persisted graph artifacts)")
    print(f"  HNSW total/video   : {hnsw_pv:.2f} B  (CLIP4Clip video-feature anchor + graph)")
    print(f"  IVF-PQ m{PQ_M} fixed : {(pq_codebook_b + ivf_centroid_b + IVFPQ_FAISS_META_BYTES)/1024:.2f} KB  (PQ codebook + IVF centroids + FAISS metadata)")
    print(f"  OPQ m{PQ_M} fixed    : {(pq_codebook_b + ivf_centroid_b + opq_rotation_b + OPQ_FAISS_META_BYTES)/1024:.2f} KB  (IVF-PQ fixed + OPQ rotation)")

    rows = []  # (method, N, component, bytes)
    for N in NS:
        # GRDR per-video index footprint (Storage Accounting SSOT): V*L codes + int64 id.
        rows.append(("GRDR", N, "sid_payload", GRDR_V * GRDR_L * GRDR_B_CODE * N))
        rows.append(("GRDR", N, "video_id", GRDR_B_ID * N))
        rows.append(("frame_dense", N, "features", frame_pv * N))
        rows.append(("video_dense", N, "features", video_pv * N))
        rows.append(("hnsw", N, "features", video_pv * N))
        rows.append(("hnsw", N, "graph", hnsw_graph_pv * N))
        rows.append(("ivfpq", N, "pq_codes", PQ_M * N))
        rows.append(("ivfpq", N, "video_id", ANN_B_ID * N))
        rows.append(("ivfpq", N, "pq_codebook", pq_codebook_b))
        rows.append(("ivfpq", N, "ivf_centroids", ivf_centroid_b))
        rows.append(("ivfpq", N, "faiss_metadata", IVFPQ_FAISS_META_BYTES))
        rows.append(("opq", N, "pq_codes", PQ_M * N))
        rows.append(("opq", N, "video_id", ANN_B_ID * N))
        rows.append(("opq", N, "pq_codebook", pq_codebook_b))
        rows.append(("opq", N, "ivf_centroids", ivf_centroid_b))
        rows.append(("opq", N, "opq_rotation", opq_rotation_b))
        rows.append(("opq", N, "faiss_metadata", OPQ_FAISS_META_BYTES))
        rows.append(("ivf", N, "vectors", vec_pv * N))
        rows.append(("ivf", N, "video_id", ANN_B_ID * N))

        assert sum(b for m, n, _, b in rows if m == "ivfpq" and n == N) == ivfpq_bytes(N, DIM, PQ_M, PQ_NBITS, IVF_NLIST)
        assert sum(b for m, n, _, b in rows if m == "opq" and n == N) == opq_bytes(N, DIM, PQ_M, PQ_NBITS, IVF_NLIST)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("method,N,component,bytes\n")
        for m, N, c, b in rows:
            f.write(f"{m},{N},{c},{b:.0f}\n")
    print(f"\nwrote {len(rows)} rows to {args.out}")

    # totals table
    tot = defaultdict(lambda: defaultdict(float))
    for m, N, c, b in rows:
        tot[m][N] += b
    print("\nmethod totals (MB):")
    table_methods = ["GRDR", "ivfpq", "opq", "video_dense", "hnsw", "ivf", "frame_dense"]
    print("  N        " + "  ".join(f"{m:>12}" for m in table_methods))
    for N in NS:
        line = f"  {N:<8}"
        for m in table_methods:
            line += f"  {tot[m][N]/1e6:>12.1f}"
        print(line)


if __name__ == "__main__":
    main()
