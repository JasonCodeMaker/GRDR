#!/usr/bin/env python
"""Convert GRDR pseudo text-feature pickle to remapped .npy memmap artifacts."""

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np


def load_pickle(path, label):
    start = time.time()
    print(f"[load] {label}: {path}", flush=True)
    with path.open('rb') as f:
        data = pickle.load(f)
    print(f"[load] {label}: {len(data):,} entries in {time.time() - start:.1f}s", flush=True)
    return data


def file_stat(path):
    st = path.stat()
    return {
        'path': str(path),
        'name': path.name,
        'size': st.st_size,
        'mtime_ns': st.st_mtime_ns,
    }


def compute_original_counts(train_text):
    counts = {}
    for key in train_text.keys():
        video_id, suffix = key.rsplit('_', 1)
        if suffix.startswith('a'):
            continue
        counts[video_id] = max(counts.get(video_id, 0), int(suffix) + 1)
    return counts


def remap_pseudo_key(key, original_counts):
    video_id, suffix = key.rsplit('_', 1)
    idx_str = suffix[1:] if suffix.startswith('a') else suffix
    return f"{video_id}_{original_counts.get(video_id, 0) + int(idx_str)}"


def remove_if_exists(path):
    if path.exists():
        path.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'pseudo_pkl',
        type=Path,
        help='Path to text_embeddings_train_addition.pkl',
    )
    parser.add_argument(
        '--train-text-pkl',
        type=Path,
        default=None,
        help='Path to text_embeddings_train.pkl; defaults to sibling file.',
    )
    parser.add_argument(
        '--verify-samples',
        type=int,
        default=32,
        help='Number of deterministic random rows to verify after conversion.',
    )
    parser.add_argument(
        '--progress-every',
        type=int,
        default=500000,
        help='Print conversion progress every N rows.',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing memmap artifacts.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pseudo_pkl = args.pseudo_pkl.resolve()
    train_text_pkl = (
        args.train_text_pkl.resolve()
        if args.train_text_pkl is not None
        else pseudo_pkl.with_name('text_embeddings_train.pkl')
    )
    if not pseudo_pkl.exists():
        raise FileNotFoundError(pseudo_pkl)
    if not train_text_pkl.exists():
        raise FileNotFoundError(train_text_pkl)

    out_npy = pseudo_pkl.with_suffix('.npy')
    out_idx = pseudo_pkl.with_suffix('.idx.json')
    out_meta = Path(str(out_npy) + '.meta.json')
    final_paths = (out_npy, out_idx, out_meta)
    if any(p.exists() for p in final_paths):
        if not args.force:
            existing = ', '.join(str(p) for p in final_paths if p.exists())
            raise FileExistsError(f"Refusing to overwrite existing artifacts: {existing}")
        for p in final_paths:
            remove_if_exists(p)

    tmp_npy = Path(str(out_npy) + '.tmp')
    tmp_idx = Path(str(out_idx) + '.tmp')
    tmp_meta = Path(str(out_meta) + '.tmp')
    for p in (tmp_npy, tmp_idx, tmp_meta):
        remove_if_exists(p)

    train_text = load_pickle(train_text_pkl, 'train text')
    original_counts = compute_original_counts(train_text)
    train_text_entries = len(train_text)
    print(f"[counts] {len(original_counts):,} videos from {train_text_entries:,} original text keys", flush=True)
    del train_text

    pseudo = load_pickle(pseudo_pkl, 'pseudo text')
    keys = list(pseudo.keys())
    sample = next(iter(pseudo.values()))
    shape = (len(keys), *sample.shape)
    dtype = sample.dtype
    print(f"[write] {out_npy} shape={shape} dtype={dtype}", flush=True)
    arr = np.lib.format.open_memmap(tmp_npy, mode='w+', dtype=dtype, shape=shape)

    idx = {}
    start = time.time()
    for i, key in enumerate(keys):
        new_key = remap_pseudo_key(key, original_counts)
        if new_key in idx:
            raise ValueError(f"Duplicate remapped key: {new_key}")
        idx[new_key] = i
        arr[i] = pseudo[key]
        if args.progress_every > 0 and (i + 1) % args.progress_every == 0:
            elapsed = time.time() - start
            print(f"[write] {i + 1:,}/{len(keys):,} rows in {elapsed:.1f}s", flush=True)
    arr.flush()
    del arr

    print(f"[write] {out_idx}", flush=True)
    with tmp_idx.open('w') as f:
        json.dump(idx, f, separators=(',', ':'))

    print(f"[verify] {args.verify_samples} deterministic random rows", flush=True)
    mm = np.load(tmp_npy, mmap_mode='r')
    random.seed(42)
    for key in random.sample(keys, min(args.verify_samples, len(keys))):
        new_key = remap_pseudo_key(key, original_counts)
        if not np.array_equal(pseudo[key], mm[idx[new_key]]):
            raise AssertionError(f"Round-trip mismatch for {key} -> {new_key}")

    meta = {
        'format_version': 1,
        'key_mode': 'remapped',
        'array_file': out_npy.name,
        'index_file': out_idx.name,
        'shape': list(shape),
        'dtype': str(dtype),
        'num_entries': len(keys),
        'train_text_entries': train_text_entries,
        'sources': {
            'pseudo_pickle': file_stat(pseudo_pkl),
            'train_text_pickle': file_stat(train_text_pkl),
        },
        'created_at_unix': time.time(),
        'verify_samples': min(args.verify_samples, len(keys)),
    }
    with tmp_meta.open('w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    os.replace(tmp_npy, out_npy)
    os.replace(tmp_idx, out_idx)
    os.replace(tmp_meta, out_meta)
    print("[done] wrote:", flush=True)
    for p in final_paths:
        print(f"  {p} ({p.stat().st_size:,} bytes)", flush=True)


if __name__ == '__main__':
    main()
