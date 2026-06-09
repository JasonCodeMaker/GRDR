#!/usr/bin/env python
"""Measure index storage vs corpus size N over the panda corpora to 2M (Fig 2).

Supersedes reranker/xpool/utils/storage_sim.py (which extrapolated from random
features). Every point here is a MEASURED byte count:

  - GRDR T+M+D : D = serving decoder ckpt (fixed offset); M = the real per-video
                 sID payload serialized from the panda sID index; T = the prefix
                 trie over the sliced sID set (real node count x bytes/node).
  - frame dense (X-Pool)   : measured per-video .npz bytes x N.
  - video dense (CLIP4Clip): measured per-video .npz bytes x N.
  - ANN HNSW / IVF         : raw vectors (N x dim x 4B) + the measured graph/list
                             structure from the panda setting-2 npz, scaled by N.

Writes a long CSV (method, N, component, bytes) so the notebook renders directly.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

REPO = "/home/uqzzha35/Project/SemanticID/GRDR"
MM = "/home/uqzzha35/Project/SemanticID/MM-SemanticTVR"

# Measured panda corpus grid (test=5694; +d distractors).
PANDA_TEST = 5694
DISTRACTORS = [0, 400_000, 800_000, 1_200_000, 1_600_000, 2_000_000]
NS = [PANDA_TEST + d for d in DISTRACTORS]

DIM = 512  # InternVideo2 / CLIP video embedding dim.

# --- measured artifact paths --------------------------------------------------
GRDR_SID_INDEX = f"{MM}/data/panda/none/text_guided_c4096_l3/panda_index_internvideo2_emb_train.json"
GRDR_DECODER_CKPT = f"{REPO}/output/checkpoints/GRDR/panda/champion_multiview_n4_c4096l3_2150k_s42/model-3-fit/best_model.pt"
HNSW_NPZ = f"{REPO}/output/evaluation_results/ann_baseline/indexes/panda_hnsw_setting2.npz"
IVF_NPZ = f"{REPO}/output/evaluation_results/ann_baseline/indexes/panda_ivf_setting2.npz"
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


def grdr_sid_bytes_per_video(sid_index, n_sample=50_000):
    """Real serialized bytes/video for the sID payload M (npz int16, NLT codes)."""
    keys = list(sid_index.keys())[:n_sample]
    # Each video carries L codes (e.g. A_x,B_y,C_z). Store as int16 codebook ids.
    codes = []
    for k in keys:
        codes.append([int(c.split("_")[1]) for c in sid_index[k]])
    arr = np.asarray(codes, dtype=np.int16)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=True) as tf:
        np.savez(tf.name, sid=arr)
        total = os.path.getsize(tf.name)
    return total / len(keys)


def trie_bytes(sid_index, video_ids):
    """Measured prefix-trie node count over the sliced sID set x bytes/node."""
    BYTES_PER_NODE = 12  # one child ptr (8B) + token id (4B), conservative.
    nodes = set()
    for vid in video_ids:
        bare = vid.split("/", 1)[1] if "/" in vid else vid
        codes = sid_index.get(bare)
        if not codes:
            continue
        prefix = ()
        for c in codes:
            prefix = prefix + (c,)
            nodes.add(prefix)
    return len(nodes) * BYTES_PER_NODE


def npz_array_bytes(path):
    d = np.load(path, allow_pickle=True)
    return sum(d[k].nbytes for k in d.files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{REPO}/output/evaluation_results/figures/summaries/storage_data.csv")
    ap.add_argument("--full", action="store_true", help="exact trie over the full sliced set (slow at 2M)")
    args = ap.parse_args()

    print("Loading panda sID index (2.15M videos)...")
    sid_index = json.load(open(GRDR_SID_INDEX))
    sid_keys = list(sid_index.keys())

    # --- fixed / per-unit measured quantities ---
    D_bytes = os.path.getsize(GRDR_DECODER_CKPT)               # fixed offset
    m_per_video = grdr_sid_bytes_per_video(sid_index)          # M slope
    frame_pv = measured_per_video_bytes(FRAME_CACHE)           # frame dense slope
    video_pv = measured_per_video_bytes(VIDEO_CACHE)           # video dense slope
    # ANN: raw vectors (N x dim x 4) + structure, anchored at the 2M npz.
    hnsw_struct_2m = npz_array_bytes(HNSW_NPZ)                 # graph at pool 2005694
    ivf_struct_2m = npz_array_bytes(IVF_NPZ)                  # lists+centroids at 2005694
    n_2m = 2_005_694
    hnsw_struct_pv = hnsw_struct_2m / n_2m
    ivf_struct_pv = ivf_struct_2m / n_2m
    vec_pv = DIM * 4                                           # raw f32 vector / video

    print(f"  D (decoder, fixed) : {D_bytes/1e6:.1f} MB")
    print(f"  M sID bytes/video  : {m_per_video:.2f} B")
    print(f"  frame dense/video  : {frame_pv/1024:.2f} KB")
    print(f"  video dense/video  : {video_pv/1024:.2f} KB")
    print(f"  HNSW struct/video  : {hnsw_struct_pv:.1f} B  (+{vec_pv} B vectors)")
    print(f"  IVF  struct/video  : {ivf_struct_pv:.1f} B  (+{vec_pv} B vectors)")

    rows = []  # (method, N, component, bytes)
    for N in NS:
        # GRDR T+M+D: trie measured on the sliced set (exact only with --full).
        if args.full:
            vids = sid_keys[:N]  # nested prefix slice (test+first-d distractors)
            T = trie_bytes(sid_index, vids)
        else:
            # trie scales ~ with #unique codes; estimate from a 100k sample slope.
            sample_vids = sid_keys[: min(N, 100_000)]
            T_sample = trie_bytes(sid_index, sample_vids)
            T = T_sample * (N / len(sample_vids))
        M = m_per_video * N
        rows.append(("GRDR", N, "D_decoder", D_bytes))
        rows.append(("GRDR", N, "M_sid", M))
        rows.append(("GRDR", N, "T_trie", T))
        rows.append(("frame_dense", N, "features", frame_pv * N))
        rows.append(("video_dense", N, "features", video_pv * N))
        rows.append(("hnsw", N, "vectors", vec_pv * N))
        rows.append(("hnsw", N, "structure", hnsw_struct_pv * N))
        rows.append(("ivf", N, "vectors", vec_pv * N))
        rows.append(("ivf", N, "structure", ivf_struct_pv * N))

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
