# GRDR Storage Accounting — Project SSOT

Single source of truth for how **per-video index storage** is computed for GRDR and every
baseline, across all packages, figures, and the paper. Adopted 2026-06-09. Supersedes the
earlier `D_decoder + M_sid + T_trie` accounting in `storage_measure.py` and the
`2026-06-09-storage-matched-ann` result tables (which counted a fixed 275 MB decoder and a
single-view 6 B/vid sID payload — both wrong; see "What changed").

## Convention

Storage = the **per-video index footprint** required to serve stage-1 retrieval, accounted
**symmetrically** across methods. The rule for what counts:

- **Count** the per-video payload (codes), the learned **dictionary** consulted at query time,
  and the **video-id** mapping needed to return results.
- **Exclude**, on every method, the **corpus-independent query/tokenizer models** and any
  **rebuildable search structure**. These are excluded because they appear on *both* sides and
  cancel — counting them for one method only is the asymmetry this SSOT removes.

| component | GRDR | ANN (IVF-PQ / OPQ) | counted? | why |
|---|---|---|---|---|
| query-side encoder | T5 generator (~240 MB) | fine-tuned CLIP/InternVideo text tower | **excluded (both)** | corpus-fine-tuned, N-independent, one per method → cancels |
| offline tokenizer | RQ-VAE codebooks | video encoder | **excluded (both)** | used offline to build the index; not consulted at query time |
| query-time dictionary | — (none) | PQ codebook (0.52 MB) + OPQ rotation (1 MB) | **counted** | consulted per query (ADC); GRDR has no counterpart |
| per-video payload | `V·L` sID codes | `m` PQ codes | **counted** | the actual per-video index data |
| video id | int64 in `sID→video` resolver | int64 alongside each code (IVF reorders off video order) | **counted (both)** | symmetric; same cost, same reason |
| search structure | prefix trie (constrains decode) | IVF lists / HNSW graph | **excluded (both)** | derivable from payload, rebuilt at load; negligible for IVF-PQ anyway |

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
IVF-PQ / OPQ      S(N) = N · (m + b_id) [+ codebook 0.52 MB (+ OPQ rot 1 MB)]  = (m+8)·N + tiny
IVF-Flat / HNSW   S(N) = N · (dim·4 + b_id) = 2056 · N                              (uncompressed anchors)
```

Per-video slope (the comparison axis): **GRDR 32 B/vid**; IVF-PQ m8 = 16, m16 = 24, m32 = 40,
m64 = 72 B/vid (incl. id). GRDR sits **between m16 and m32** → not the most compact index.

## Worked totals (Panda N-grid, total index MB)

| N | GRDR (32 B) | IVF-PQ m8 | IVF-PQ m16 | IVF-PQ m32 | IVF-PQ m64 |
|---|---|---|---|---|---|
| 5,694 | 0.18 | 0.05 | 0.09 | 0.18 | 0.36 |
| 405,694 | 13.0 | 6.5 | 9.7 | 16.2 | 29.2 |
| 805,694 | 25.8 | 12.9 | 19.3 | 32.2 | 58.0 |
| 1,205,694 | 38.6 | 19.3 | 28.9 | 48.2 | 86.8 |
| 1,605,694 | 51.4 | 25.7 | 38.5 | 64.2 | 115.6 |
| 2,005,694 | 64.2 | 32.1 | 48.1 | 80.2 | 144.4 |

GRDR's per-video slope (32 B) exceeds every recall-preserving PQ budget (m8=16, m16=24), so PQ
is more compact at **every** N. There is no fixed-decoder intercept and no large-N crossover.

## What changed vs the original `storage_measure.py`

1. **Dropped the 275 MB `D_decoder`** — the T5 generator is the query-side model and cancels
   against ANN's fine-tuned text encoder (both corpus-trained, N-independent).
2. **Dropped `T_trie`** — rebuildable from the sID set at load; symmetric with ANN's
   IVF/HNSW structure, which is negligible for IVF-PQ.
3. **Fixed the multi-view undercount** — count all `V=4` sIDs (24 B codes), not the single-view
   6 B from the MM json.

Net: GRDR's reported per-video storage moves from a decoder-dominated 275–319 MB absolute (or a
single-view ~22 B/vid marginal) to a clean **32 B/vid** linear footprint. The
`storage-matched-ann` honest-negative is **preserved** (GRDR still above the recall-preserving
PQ budget); only the framing is corrected (no decoder penalty, no 119M crossover).
