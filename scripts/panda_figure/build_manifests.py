#!/usr/bin/env python
"""Build NESTED distractor-pool manifests for the Panda pool-scaling figure.

One seed-42 shuffle of the unique Panda train video IDs is computed once; pool
d = test ∪ shuffle[:d]. Because every manifest is a prefix of the same shuffle,
the pools nest (400k ⊂ 800k ⊂ … ⊂ 2.0M), so every method sees the *same* videos
accumulating and the per-method curves are directly comparable.

Each distractor count d writes:
  <out_dir>/panda_pool_d<d>.json
    = {"pool_size": N_TEST+d, "n_test": N_TEST, "n_distractors": d, "seed": 42,
       "video_ids": [<distractor ids, no .mp4>]}   # distractors only

This schema is consumed verbatim by reranker/xpool/test.py via
  --panda_distractor_manifest <path> --expanded_pool
(data_factory.get_train_video_ids replaces the full 2.15M train list with these
ids; the X-Pool pool = N_TEST test embeddings + these distractor embeddings).
The ANN / generative drivers read the same files so all methods share the pool.
"""
import argparse
import json
import random
from pathlib import Path

import ujson


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_json", required=True, help="panda_ret_train.json")
    p.add_argument("--test_json", required=True, help="panda_ret_test.json")
    p.add_argument("--out_dir", required=True, help="manifest output dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distractors", type=int, nargs="+",
                   default=[0, 400_000, 800_000, 1_200_000, 1_600_000, 2_000_000])
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def unique_video_ids(rows):
    """Strip .mp4, dedup preserving first-seen order."""
    seen, out = set(), []
    for row in rows:
        vid = row["video"].replace(".mp4", "")
        if vid not in seen:
            seen.add(vid)
            out.append(vid)
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.test_json) as f:
        test_ids = unique_video_ids(ujson.load(f))
    with open(args.train_json) as f:
        train_ids = unique_video_ids(ujson.load(f))
    print(f"n_test={len(test_ids):,}  train_uniques={len(train_ids):,}")

    # ONE shuffle; prefixes give nested pools.
    shuffled = list(train_ids)
    random.Random(args.seed).shuffle(shuffled)

    summary = {"seed": args.seed, "n_test": len(test_ids),
               "train_uniques": len(train_ids), "legs": []}

    for d in args.distractors:
        clamped = min(d, len(train_ids))
        if clamped < d:
            print(f"WARN: d={d:,} > train uniques; clamping distractors to {clamped:,}")
        distractor_ids = shuffled[:clamped]
        pool_size = len(test_ids) + clamped
        out_file = out_dir / f"panda_pool_d{d}.json"
        if out_file.exists() and not args.force:
            print(f"  skip existing {out_file}")
        else:
            with open(out_file, "w") as f:
                json.dump({
                    "pool_size": pool_size,
                    "n_test": len(test_ids),
                    "n_distractors": clamped,
                    "seed": args.seed,
                    "video_ids": distractor_ids,
                }, f)
            print(f"  wrote {out_file}  (pool_size={pool_size:,}, n_distractors={clamped:,})")
        summary["legs"].append({"distractor_n": d, "pool_size": pool_size,
                                "n_distractors": clamped, "path": str(out_file)})

    summary_path = out_dir / "panda_pool_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
