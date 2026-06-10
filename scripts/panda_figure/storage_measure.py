#!/usr/bin/env python
"""Measure index storage vs corpus size N over the panda corpora to 2M (Fig 2).

Supersedes reranker/xpool/utils/storage_sim.py (which extrapolated from random
features). GRDR follows the project Storage Accounting SSOT; the baselines are
measured byte counts:

  - GRDR        : per-video index footprint per the Storage Accounting SSOT
                 (scripts/panda_figure/storage_accounting_ssot.md): V*L sID codes +
                 int64 video id = 32 B/vid. The T5 decoder, RQ-VAE codebooks, and the
                 prefix trie are EXCLUDED (query-side / offline / rebuildable; they
                 cancel against ANN's encoder / structure — see the SSOT).
  - frame dense (X-Pool)   : measured per-video .npz bytes x N.
  - video dense (CLIP4Clip): measured per-video .npz bytes x N.
  - ANN IVF-Flat / HNSW    : raw vectors (N x dim x 4B) + int64 id, per the Storage
                             Accounting SSOT; the IVF lists / HNSW graph are EXCLUDED
                             (rebuildable at load, symmetric with GRDR's excluded trie),
                             so both anchors are 2048 + 8 = 2056 B/video.

Writes a long CSV (method, N, component, bytes) so the notebook renders directly.
"""
import argparse
import os
from collections import defaultdict

import numpy as np

from storage_ssot import GRDR_V, GRDR_L, GRDR_B_CODE, GRDR_B_ID, ANN_B_ID, grdr_bytes_per_video

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
    vec_pv = DIM * 4                                           # raw f32 vector / video
    grdr_pv = grdr_bytes_per_video()                          # SSOT 32 B/video
    anchor_pv = vec_pv + ANN_B_ID                             # SSOT anchor: vector + int64 id (structure excluded)

    print(f"  GRDR bytes/video   : {grdr_pv} B  (SSOT: {GRDR_V}*{GRDR_L}*{GRDR_B_CODE} codes + {GRDR_B_ID} id)")
    print(f"  frame dense/video  : {frame_pv/1024:.2f} KB")
    print(f"  video dense/video  : {video_pv/1024:.2f} KB")
    print(f"  ANN anchor/video   : {anchor_pv} B  (SSOT: {vec_pv} B vector + {ANN_B_ID} B id; IVF/HNSW structure excluded as rebuildable)")

    rows = []  # (method, N, component, bytes)
    for N in NS:
        # GRDR per-video index footprint (Storage Accounting SSOT): V*L codes + int64 id.
        rows.append(("GRDR", N, "sid_payload", GRDR_V * GRDR_L * GRDR_B_CODE * N))
        rows.append(("GRDR", N, "video_id", GRDR_B_ID * N))
        rows.append(("frame_dense", N, "features", frame_pv * N))
        rows.append(("video_dense", N, "features", video_pv * N))
        rows.append(("hnsw", N, "vectors", vec_pv * N))
        rows.append(("hnsw", N, "video_id", ANN_B_ID * N))
        rows.append(("ivf", N, "vectors", vec_pv * N))
        rows.append(("ivf", N, "video_id", ANN_B_ID * N))

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
    print("  N        " + "  ".join(f"{m:>12}" for m in ["GRDR", "video_dense", "hnsw", "ivf", "frame_dense"]))
    for N in NS:
        line = f"  {N:<8}"
        for m in ["GRDR", "video_dense", "hnsw", "ivf", "frame_dense"]:
            line += f"  {tot[m][N]/1e6:>12.1f}"
        print(line)


if __name__ == "__main__":
    main()
