"""Project SSOT for per-video index storage — the single importable implementation
of the formulas documented in storage_accounting_ssot.md. Figures, the measurement
script, and any baseline import these instead of hardcoding bytes-per-video.
"""

# GRDR multi-view sID footprint (champion ...n4_c4096l3...; see storage_accounting_ssot.md).
GRDR_V = 4        # latent-token sIDs per video (all routes stored; trainer/evaluator.py:587-592)
GRDR_L = 3        # codes per sID (RQ code_length)
GRDR_B_CODE = 2   # bytes/code (int16; K=4096 -> 12-bit, 1.5 if bit-packed)
GRDR_B_ID = 8     # int64 video id

DIM = 512         # video embedding dim (uncompressed anchor)
ANN_B_ID = 8      # int64 id per vector (IVF reorders off video order)


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


def anchor_bytes_per_video(dim=DIM):
    """Uncompressed IVF-Flat/HNSW per-video bytes: fp32 vector + int64 id."""
    return dim * 4 + ANN_B_ID


def anchor_bytes(n, dim=DIM):
    """Total uncompressed-anchor index bytes (excludes rebuildable graph/list structure)."""
    return anchor_bytes_per_video(dim) * n
