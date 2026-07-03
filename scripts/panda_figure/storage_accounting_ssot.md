# GRDR Storage Accounting — Project SSOT

Single source of truth for how **per-video index storage** is computed for GRDR and every
baseline, across all packages, figures, and the paper. Adopted 2026-06-09. Supersedes the
earlier `D_decoder + M_sid + T_trie` accounting in `storage_measure.py` and the
`2026-06-09-storage-matched-ann` result tables (which counted a fixed 275 MB decoder and a
single-view 6 B/vid sID payload — both wrong; see "What changed").

## Convention

Storage = the **per-video index footprint** required to serve stage-1 retrieval, accounted
**symmetrically** across methods. The rule for what counts:

- **Count** the per-video payload (codes/features), the learned **dictionary** consulted at query
  time, the **video-id** mapping needed to return results, and any persisted method-specific
  serving structure reported as part of the index artifact.
- **Exclude**, on every method, the **corpus-independent query/tokenizer models** and structures
  that are not persisted in the reported serving artifact. The GRDR prefix trie remains excluded:
  it is rebuilt from the Semantic ID payload at load time and is not stored as a separate index
  artifact.

| component | GRDR | ANN (IVF-PQ / OPQ) | counted? | why |
|---|---|---|---|---|
| query-side encoder | T5 generator (~240 MB) | fine-tuned CLIP/InternVideo text tower | **excluded (both)** | corpus-fine-tuned, N-independent, one per method → cancels |
| offline tokenizer | RQ-VAE codebooks | video encoder | **excluded (both)** | used offline to build the index; not consulted at query time |
| query-time dictionary / quantizer | — (none) | PQ codebook (0.52 MB) + IVF coarse centroids (8.39 MB, `nlist=4096`) + OPQ rotation (1.05 MB) | **counted** | consulted per query / stored in the FAISS index; GRDR has no counterpart |
| per-video payload | `V·L` sID codes | dense video-feature artifact / `m` PQ codes | **counted** | Fig. 2 compares HNSW against the same measured CLIP4Clip video-feature anchor, then adds graph |
| video id | int64 in `sID→video` resolver | int64 alongside each code (IVF reorders off video order) | **counted (both)** | symmetric; same cost, same reason |
| search structure | prefix trie (constrains decode) | IVF lists / HNSW graph | **GRDR trie excluded; HNSW graph counted when persisted** | Fig. 2 counts the measured persisted HNSW graph artifact; GRDR has no persisted graph counterpart |

X-Pool is **stage-2 and shared by both pipelines** (same checkpoint), so its fine-tuned CLIP is
a common downstream cost counted on neither side. The symmetry only needs to hold for stage-1.

## Verified constants (champion `…n4_c4096l3…`)

Read from the deployed retrieval code, not assumed:

- `V = 4` — sIDs (latent-token routes) per video. Each video is registered under **all 4** of
  its token-sIDs in the `sID→video` map: [`trainer/evaluator.py:587-592`](../../trainer/evaluator.py#L587-L592)
  (`for code in token_codes: corpus_ids.append([0, *code])` + `build_sid_to_videos_mapping`),
  and `gen_sid` stores `sample_codes_dict[video_id] = code` where `code` iterates `token_idx`
  over `num_latent_tokens` ([`models/grdr.py:347-355`](../../models/grdr.py#L347-L355)).
- `L = 3` — codes per sID (RQ `code_length`, the `A_/B_/C_` layers in the index json).
- `b_code = 2 B` — codebook `K=4096` → 12-bit ids, stored int16 (1.5 B if bit-packed).
- `b_id = 8 B` — int64 video id.

> The MM-SemanticTVR json `panda_index_internvideo2_emb_train.json` stores a **single-view**
> sID (3 codes = 6 B/vid). It is *not* the deployed multi-view index and must not be used to
> size GRDR storage — that was the 4× undercount in the original measurement.

## Formulas

```
GRDR              S(N) = N · (V·L·b_code + b_id) = N · (4·3·2 + 8) = 32 · N  bytes   (26·N if 12-bit packed)
IVF-PQ m16        S(N) = N · (m + b_id) + B_pq_codebook + B_ivf_centroids + B_faiss_meta
                       = 24·N + 524,288 + 8,388,608 + 32,948 bytes                  (Panda 2M faiss.serialize_index measured)
OPQ m16           S(N) = S_IVFPQ(N) + B_opq_rotation + B_opq_meta_delta
                       = 24·N + 524,288 + 8,388,608 + 1,048,576 + 33,019 bytes       (Panda 2M faiss.serialize_index measured)
CLIP4Clip anchor  S(N) = N · b_video_cache
                       = N · 2660.59 bytes                                          (measured video-feature cache artifact)
IVF-Flat          S(N) = N · (dim·4 + b_id) = 2056 · N                              (raw-vector audit row; not shown in Fig. 2 storage)
HNSW              S(N) = N · (b_video_cache + b_graph)
                       = N · (2660.59 + 100.44) = 2761.03 · N bytes                 (same CLIP4Clip feature anchor + measured persisted HNSW graph)
```

The IVF-PQ/OPQ formula above is the artifact-storage contract used for Fig. 2. It replaces the
older payload-only lower bound `N·(m+8)+codebook`, which undercounts actual FAISS serialized
indexes at small and medium N because it omits IVF coarse centroids and FAISS metadata. The Fig. 2
ANN run configuration uses `ivf_nlist=4096`; the measured Panda 2M serialized artifacts are:

- IVF-PQ m16: `57,082,500` bytes at `N=2,005,694` (`28.46 B/video`), stored in
  `var/research/2026-06-22-panda-pq-storage/panda_d2000000_ivfpq_m16_storage.json`.
- OPQ m16: `58,131,147` bytes at `N=2,005,694` (`28.98 B/video`), stored in
  `var/research/2026-06-22-panda-pq-storage/panda_d2000000_opq_m16_storage.json`.

For m16, the asymptotic slope is still 24 B/video, but the fixed FAISS terms matter until the
corpus is large.

## Worked totals (Panda N-grid, total index MB)

| N | GRDR (32 B) | IVF-PQ m16 (FAISS artifact) | OPQ m16 (FAISS artifact) |
|---|---:|---:|---:|
| 5,694 | 0.18 | 9.08 | 10.13 |
| 405,694 | 13.0 | 18.7 | 19.7 |
| 805,694 | 25.8 | 28.3 | 29.3 |
| 1,205,694 | 38.6 | 37.9 | 38.9 |
| 1,605,694 | 51.4 | 47.5 | 48.5 |
| 2,005,694 | 64.2 | 57.1 | 58.1 |

Under this artifact-storage contract, IVF-PQ m16 is larger than GRDR below about 1.12M videos;
OPQ m16 is larger below about 1.25M videos. At Panda 2.0M, both m16 PQ baselines are smaller than
GRDR in storage, but they are now counted with the same serialized-index artifact semantics used
by the measured FAISS runs.

## What changed vs the original `storage_measure.py`

1. **Dropped the 275 MB `D_decoder`** — the T5 generator is the query-side model and cancels
   against ANN's fine-tuned text encoder (both corpus-trained, N-independent).
2. **Dropped `T_trie`** — rebuildable from the Semantic ID set at load and not persisted as a
   separate serving artifact. Fig. 2 now counts HNSW's persisted graph artifact when plotting HNSW.
3. **Fixed the multi-view undercount** — count all `V=4` sIDs (24 B codes), not the single-view
   6 B from the MM json.

Net: GRDR's reported per-video storage moves from a decoder-dominated 275–319 MB absolute (or a
single-view ~22 B/vid marginal) to a clean **32 B/vid** linear footprint. The
`storage-matched-ann` honest-negative is **preserved** (GRDR still above the recall-preserving
PQ budget); only the framing is corrected (no decoder penalty, no 119M crossover).
