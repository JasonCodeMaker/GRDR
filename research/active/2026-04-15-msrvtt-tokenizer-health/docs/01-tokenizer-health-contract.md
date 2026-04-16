# Tokenizer Health Contract

## Purpose

Decide whether the current VideoRQVAE tokenizer behaves like a healthy multi-slot tokenizer or already shows measurable slot redundancy / partial collapse.

## Inputs

- One unique pooled video feature per video.
- Checkpointed `VideoRQVAE_V2` weights from `best_model.pt.videorqvae`.
- Shapes:
  - video feature: `[B, 512]`
  - encoded tokens: `[B, num_latent_tokens, e_dim]`
  - discrete indices: `[B, num_latent_tokens, code_length]`

The analysis unit is `unique video`, not caption instance.

## Metrics

### 1. `encoder_pairwise_cosine`

- Meaning: average cosine similarity between different slot embeddings before quantization.
- Numerator: sum of cosine values over all within-video slot pairs.
- Denominator: number of within-video slot pairs times number of videos.
- High value means slots are geometrically similar before quantization.

### 2. `quantized_pairwise_cosine`

- Same as above, but computed on quantized slot embeddings.
- High value means quantization did not separate the slots much.

### 3. `duplicate_last_code_video_rate`

- Meaning: fraction of videos where at least two slots share the same last-layer code.
- Numerator: number of videos with any duplicate last-layer code among slots.
- Denominator: number of unique videos.

### 4. `duplicate_full_code_video_rate`

- Meaning: fraction of videos where at least two slots share the same full multi-layer code tuple.
- Numerator: number of videos with any duplicate full code tuple among slots.
- Denominator: number of unique videos.

### 5. `slot_active_codes`

- Meaning: for a fixed RQ layer and slot position, how many distinct codes are actually used across videos.
- Low value means that slot has collapsed onto a narrow subset of the codebook.

### 6. `slot_top_share`

- Meaning: maximum empirical frequency of a single code in a fixed slot and layer.
- High value means one code dominates that slot.

## Counting Rules

- Each raw video contributes once.
- Caption multiplicity does not count.
- Missing splits are skipped rather than imputed.
- Duplicate detection is exact equality on integer code IDs.

## Toy Examples

### Example A

- Slot last-layer codes for one video: `[1, 2, 3, 4]`
- `duplicate_last_code_video_rate` contribution: `0`
- `mean_unique_last_codes_per_video` contribution: `4`

### Example B

- Slot last-layer codes for one video: `[1, 1, 2, 3]`
- `duplicate_last_code_video_rate` contribution: `1`
- `mean_unique_last_codes_per_video` contribution: `3`

### Example C

- Full codes for one video:
  - slot 0: `(4, 9, 12)`
  - slot 1: `(4, 9, 12)`
  - slot 2: `(7, 1, 5)`
  - slot 3: `(8, 2, 6)`
- `duplicate_full_code_video_rate` contribution: `1`

### Example D

- Layer-0 slot-1 codes over four videos: `[8, 8, 8, 3]`
- `slot_active_codes = 2`
- `slot_top_share = 3 / 4 = 0.75`

## Edge Cases

- If the same video appears with multiple captions, it must still count once in this analysis. Otherwise caption multiplicity can fake collapse.
- Low `duplicate_full_code_video_rate` does not imply a healthy tokenizer if early layers or fixed slots already show very low active-code counts.
- High reconstruction cosine does not prove healthy slot specialization because mean reconstruction can still succeed under redundant slots.
