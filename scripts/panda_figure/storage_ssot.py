"""Project SSOT for per-video index storage — the single importable implementation
of the formulas documented in storage_accounting_ssot.md. Figures, the measurement
script, and any baseline import these instead of hardcoding bytes-per-video.
"""
import json
from pathlib import Path

# GRDR multi-view sID footprint (champion ...n4_c4096l3...; see storage_accounting_ssot.md).
GRDR_V = 4        # latent-token sIDs per video (all routes stored; trainer/evaluator.py:587-592)
GRDR_L = 3        # codes per sID (RQ code_length)
GRDR_B_CODE = 2   # bytes/code (int16; K=4096 -> 12-bit, 1.5 if bit-packed)
GRDR_B_ID = 8     # int64 video id

DIM = 512         # video embedding dim (uncompressed anchor)
ANN_B_ID = 8      # int64 id per vector (IVF reorders off video order)
IVF_NLIST = 4096  # Fig. 2 ANN baselines use --ivf_nlist 4096
PQ_M = 16         # Fig. 2 IVF-PQ/OPQ baseline: m=16, nbits=8
PQ_NBITS = 8

# FAISS serialized-index fixed metadata measured on the Panda 2M Fig. 2 pool
# (N=2,005,694, nlist=4096) via faiss.serialize_index(index).nbytes.
IVFPQ_FAISS_META_BYTES = 32_948
OPQ_FAISS_META_BYTES = 33_019


def grdr_bytes_per_video():
    """SSOT per-video bytes for the GRDR multi-view sID index (= 32 B)."""
    return GRDR_V * GRDR_L * GRDR_B_CODE + GRDR_B_ID


def grdr_bytes(n):
    """Total GRDR index bytes for a corpus of n videos."""
    return grdr_bytes_per_video() * n


def ann_bytes_per_video(m):
    """IVF-PQ / OPQ per-video bytes: m PQ-code bytes + int64 id."""
    return m + ANN_B_ID


def ann_bytes(n, m, codebook_bytes=0):
    """Total IVF-PQ/OPQ index bytes (codebook + OPQ rotation passed via codebook_bytes)."""
    return ann_bytes_per_video(m) * n + codebook_bytes


def pq_codebook_bytes(dim=DIM, m=PQ_M, nbits=PQ_NBITS):
    """FAISS PQ codebook bytes: m * 2**nbits centroids of dim/m float32 values."""
    return m * (1 << nbits) * (dim // m) * 4


def ivf_coarse_centroid_bytes(dim=DIM, nlist=IVF_NLIST):
    """FAISS IVF coarse quantizer centroid bytes."""
    return nlist * dim * 4


def opq_rotation_bytes(dim=DIM):
    """FAISS OPQ rotation matrix bytes."""
    return dim * dim * 4


def ivfpq_bytes(
    n,
    dim=DIM,
    m=PQ_M,
    nbits=PQ_NBITS,
    nlist=IVF_NLIST,
    faiss_meta_bytes=IVFPQ_FAISS_META_BYTES,
):
    """FAISS serialized IVF-PQ bytes: codes + ids + codebook + IVF centroids + metadata."""
    return (
        (m + ANN_B_ID) * n
        + pq_codebook_bytes(dim, m, nbits)
        + ivf_coarse_centroid_bytes(dim, nlist)
        + faiss_meta_bytes
    )


def opq_bytes(
    n,
    dim=DIM,
    m=PQ_M,
    nbits=PQ_NBITS,
    nlist=IVF_NLIST,
    faiss_meta_bytes=OPQ_FAISS_META_BYTES,
):
    """FAISS serialized OPQ+IVF-PQ bytes; OPQ adds the rotation matrix."""
    return (
        (m + ANN_B_ID) * n
        + pq_codebook_bytes(dim, m, nbits)
        + ivf_coarse_centroid_bytes(dim, nlist)
        + opq_rotation_bytes(dim)
        + faiss_meta_bytes
    )


def anchor_bytes_per_video(dim=DIM):
    """Uncompressed IVF-Flat/HNSW per-video bytes: fp32 vector + int64 id."""
    return dim * 4 + ANN_B_ID


def anchor_bytes(n, dim=DIM):
    """Total uncompressed-anchor index bytes (excludes rebuildable graph/list structure)."""
    return anchor_bytes_per_video(dim) * n


def measured_hnsw_graph_bytes_per_video(index_dir):
    """Weighted mean bytes/video for persisted HNSW graph artifacts."""
    total_bytes = 0
    total_videos = 0
    for graph_path in sorted(Path(index_dir).glob("*_hnsw_setting2.npz")):
        meta_path = graph_path.with_suffix(".meta.json")
        if not meta_path.exists():
            continue
        with meta_path.open() as f:
            meta = json.load(f)
        pool_size = int(meta.get("pool_size", 0))
        if pool_size <= 0:
            continue
        total_bytes += graph_path.stat().st_size
        total_videos += pool_size
    return float(total_bytes) / total_videos if total_videos else 0.0


def hnsw_bytes_per_video(feature_bytes_per_video, graph_bytes_per_video=0.0):
    """Measured video-feature anchor plus measured persisted HNSW graph bytes."""
    return feature_bytes_per_video + graph_bytes_per_video


def hnsw_bytes(n, feature_bytes_per_video, graph_bytes_per_video=0.0):
    """Total HNSW bytes under the Fig. 2 storage contract."""
    return hnsw_bytes_per_video(feature_bytes_per_video, graph_bytes_per_video) * n
