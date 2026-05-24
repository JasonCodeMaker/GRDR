#!/usr/bin/env python
"""Deterministically subset Panda train JSON + addition JSON by video_id."""
import argparse
import hashlib
import json
import os
import random
from collections import defaultdict


def _raw_video_id(video_path):
    """Match VideoTextDataset's raw_video_id derivation (mp4/avi suffix + dir strip)."""
    vid = video_path.replace('.mp4', '').replace('.avi', '')
    if '/' in vid:
        vid = vid.split('/')[-1]
    return vid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-train', required=True, help='Panda real-caption train JSON')
    parser.add_argument('--source-addition', required=True, help='Panda pseudo-caption addition JSON')
    parser.add_argument('--n', type=int, default=500_000, help='Number of unique videos to keep')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.source_train) as f:
        train = json.load(f)
    with open(args.source_addition) as f:
        addition = json.load(f)

    # Group by raw video_id (one row per (video, caption))
    train_by_vid = defaultdict(list)
    for item in train:
        train_by_vid[_raw_video_id(item['video'])].append(item)
    addition_by_vid = defaultdict(list)
    for item in addition:
        addition_by_vid[_raw_video_id(item['video'])].append(item)

    all_train_vids = sorted(train_by_vid.keys())
    n_take = min(args.n, len(all_train_vids))
    rng = random.Random(args.seed)
    # Keep rng.sample's list order — casting to a set would lose it and make
    # downstream JSON row order non-reproducible across runs.
    selected_list = rng.sample(all_train_vids, n_take)
    print(f'  selected {len(selected_list)} / {len(all_train_vids)} train videos (seed={args.seed})')

    sub_train = [it for v in selected_list for it in train_by_vid[v]]
    sub_addition = [it for v in selected_list for it in addition_by_vid.get(v, [])]
    missing = sum(1 for v in selected_list if v not in addition_by_vid)

    out_train = os.path.join(args.out_dir, 'panda_500k_ret_train.json')
    out_addition = os.path.join(args.out_dir, 'panda_500k_ret_train_addition.json')
    with open(out_train, 'w') as f:
        json.dump(sub_train, f)
    with open(out_addition, 'w') as f:
        json.dump(sub_addition, f)

    def _sha(path):
        h = hashlib.sha256()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    sentinel = os.path.join(args.out_dir, 'sha256.sentinel')
    with open(sentinel, 'w') as f:
        f.write(f'{_sha(out_train)}  {os.path.basename(out_train)}\n')
        f.write(f'{_sha(out_addition)}  {os.path.basename(out_addition)}\n')

    print(f'  wrote {out_train} ({len(sub_train)} rows)')
    print(f'  wrote {out_addition} ({len(sub_addition)} rows)')
    print(f'  wrote {sentinel}')
    print(f'  videos missing from addition: {missing}')


if __name__ == '__main__':
    main()
